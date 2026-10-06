import array
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import ExitStack, asynccontextmanager, nullcontext
from pathlib import Path
from textwrap import dedent
from unittest.mock import ANY, AsyncMock, MagicMock

import anyio
import httpx
import pytest
import readchar
import respx
import uv
from anyio.abc import SocketAttribute, SocketStream
from exceptiongroup import ExceptionGroup

from prefect import flow
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.actions import WorkPoolCreate
from prefect.events.clients import PrefectEventSubscriber
from prefect.events.filters import EventFilter, EventNameFilter, EventResourceFilter
from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_WORKER_PREFETCH_SECONDS,
    get_current_settings,
    temporary_settings,
)
from prefect.testing.cli import invoke_and_assert
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.processutils import open_process
from prefect.workers.base import BaseJobConfiguration, BaseWorker

pytestmark = [pytest.mark.usefixtures("asserting_events_worker"), pytest.mark.clear_db]


class MockKubernetesWorker(BaseWorker):
    type = "kubernetes-test"
    job_configuration = BaseJobConfiguration

    async def run(self):
        pass


@pytest.fixture
def interactive_console(monkeypatch):
    monkeypatch.setattr("prefect.cli._app.is_interactive", lambda: True)

    # `readchar` does not like the fake stdin provided by typer isolation so we provide
    # a version that does not require a fd to be attached
    def readchar():
        sys.stdin.flush()
        position = sys.stdin.tell()
        if not sys.stdin.read():
            print("TEST ERROR: CLI is attempting to read input but stdin is empty.")
            raise SystemExit(-2)
        else:
            sys.stdin.seek(position)
        return sys.stdin.read(1)

    monkeypatch.setattr("readchar._posix_read.readchar", readchar)


@pytest.fixture
async def kubernetes_work_pool(prefect_client: PrefectClient):
    work_pool = await prefect_client.create_work_pool(
        work_pool=WorkPoolCreate(name="test-k8s-work-pool", type="kubernetes-test")
    )

    with respx.mock(
        assert_all_mocked=False, base_url=PREFECT_API_URL.value(), using="httpx"
    ) as respx_mock:
        respx_mock.get("/csrf-token", params={"client": ANY}).pass_through()
        respx_mock.route(path__startswith="/work_pools/").pass_through()
        respx_mock.get("/collections/views/aggregate-worker-metadata").mock(
            return_value=httpx.Response(
                200,
                json={
                    "prefect": {
                        "prefect-agent": {
                            "type": "prefect-agent",
                            "default_base_job_configuration": {},
                        }
                    },
                    "prefect-kubernetes": {
                        "kubernetes-test": {
                            "type": "kubernetes-test",
                            "default_base_job_configuration": {},
                        }
                    },
                },
            )
        )

        yield work_pool


@pytest.fixture
def mock_worker(monkeypatch):
    mock_worker_start = AsyncMock()
    mock_worker = MagicMock()
    mock_worker.return_value.start = mock_worker_start
    import prefect.cli._worker_utils

    monkeypatch.setattr(
        prefect.cli._worker_utils, "lookup_type", lambda x, y: mock_worker
    )
    return mock_worker


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_run_once_with_name():
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            "test-work-pool",
            "-n",
            "test-worker",
            "-t",
            "process",
        ],
        expected_code=0,
        expected_output_contains=[
            "Worker 'test-worker' started!",
            "Worker 'test-worker' stopped!",
        ],
    )


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_start_worker_creates_work_pool(prefect_client: PrefectClient):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            "not-yet-created-pool",
            "-t",
            "process",
        ],
        expected_code=0,
        expected_output_contains=["Worker", "stopped!", "Worker", "started!"],
    )

    work_pool = await prefect_client.read_work_pool("not-yet-created-pool")
    assert work_pool is not None
    assert work_pool.name == "not-yet-created-pool"
    assert work_pool.default_queue_id is not None


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_start_worker_creates_work_pool_with_base_config(
    prefect_client: PrefectClient,
):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "--pool",
            "my-cool-pool",
            "--type",
            "process",
            "--base-job-template",
            Path(__file__).parent / "base-job-templates" / "process-worker.json",
        ],
        expected_code=0,
        expected_output_contains=["Worker", "stopped!", "Worker", "started!"],
    )

    work_pool = await prefect_client.read_work_pool("my-cool-pool")
    assert work_pool is not None
    assert work_pool.name == "my-cool-pool"
    assert work_pool.default_queue_id is not None
    assert work_pool.base_job_template == {
        "job_configuration": {"command": "{{ command }}", "name": "{{ name }}"},
        "variables": {
            "properties": {
                "command": {
                    "description": "Command to run.",
                    "title": "Command",
                    "type": "string",
                },
                "name": {
                    "description": "Description.",
                    "title": "Name",
                    "type": "string",
                },
            },
            "type": "object",
        },
    }


@pytest.fixture
def unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    async def raise_connect_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(PrefectClient, "read_work_pool", raise_connect_error)
    monkeypatch.setattr(PrefectClient, "read_work_queues", raise_connect_error)


@pytest.mark.usefixtures("use_hosted_api_server", "unreachable_api")
def test_start_worker_when_api_is_unreachable(mock_worker: MagicMock):
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-p",
            "test-work-pool",
            "-t",
            "process",
            "--run-once",
        ],
        expected_code=0,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.parametrize(
    ("status_code", "should_recommend_type"),
    [
        pytest.param(None, True, id="transport-error"),
        pytest.param(503, True, id="server-error"),
        pytest.param(401, False, id="authentication-error"),
        pytest.param(403, False, id="authorization-error"),
    ],
)
@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_without_type_when_api_request_fails(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int | None,
    should_recommend_type: bool,
):
    request = httpx.Request("GET", "https://api.prefect.io/work_pools/test-work-pool")
    api_error: httpx.HTTPError
    if status_code is None:
        api_error = httpx.ConnectError(
            "All connection attempts failed", request=request
        )
    else:
        message = {
            401: "Unauthorized",
            403: "Forbidden",
            503: "Service unavailable",
        }[status_code]
        api_error = httpx.HTTPStatusError(
            message,
            request=request,
            response=httpx.Response(status_code, request=request),
        )

    async def raise_api_error(*args: object, **kwargs: object) -> None:
        raise api_error

    monkeypatch.setattr(PrefectClient, "read_work_pool", raise_api_error)
    monkeypatch.setattr(PrefectClient, "read_work_queues", raise_api_error)

    invoke_and_assert(
        command=["worker", "start", "-p", "test-work-pool", "--run-once"],
        expected_code=1,
        expected_output_contains=[
            str(api_error),
            *(["Provide a worker type with '--type'"] if should_recommend_type else []),
        ],
        expected_output_does_not_contain=(
            None if should_recommend_type else "Provide a worker type with '--type'"
        ),
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_work_queue_names(mock_worker, process_work_pool):
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-p",
            process_work_pool.name,
            "--work-queue",
            "a",
            "-q",
            "b",
            "--run-once",
        ],
        expected_code=0,
    )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name=process_work_pool.name,
        work_queues=["a", "b"],
        prefetch_seconds=ANY,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_specified_work_queues_paused(mock_worker, process_work_pool):
    invoke_and_assert(
        command=[
            "work-queue",
            "pause",
            "default",
            "--pool",
            process_work_pool.name,
        ],
        expected_code=0,
        expected_output_contains=[
            f"Work queue 'default' in work pool {process_work_pool.name!r} paused"
        ],
    )

    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-p",
            process_work_pool.name,
            "--work-queue",
            "default",
            "--run-once",
        ],
        expected_code=0,
        expected_output_contains=[
            f"Specified work queue(s) in the work pool {process_work_pool.name!r} are currently paused.",
        ],
    )

    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name=process_work_pool.name,
        work_queues=["default"],
        prefetch_seconds=ANY,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_all_work_queues_paused(mock_worker, process_work_pool):
    invoke_and_assert(
        command=[
            "work-queue",
            "pause",
            "default",
            "--pool",
            process_work_pool.name,
        ],
        expected_code=0,
        expected_output_contains=[
            f"Work queue 'default' in work pool {process_work_pool.name!r} paused"
        ],
    )

    invoke_and_assert(
        command=["worker", "start", "-p", process_work_pool.name, "--run-once"],
        expected_code=0,
        expected_output_contains=[
            f"All work queues in the work pool {process_work_pool.name!r} are currently paused.",
        ],
    )

    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name=process_work_pool.name,
        work_queues=None,
        prefetch_seconds=ANY,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_prefetch_seconds(mock_worker):
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "--prefetch-seconds",
            "30",
            "-p",
            "test",
            "--run-once",
            "-t",
            "process",
        ],
        expected_code=0,
    )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name="test",
        work_queues=None,
        prefetch_seconds=30,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_prefetch_seconds_from_setting_by_default(mock_worker):
    with temporary_settings({PREFECT_WORKER_PREFETCH_SECONDS: 100}):
        invoke_and_assert(
            command=[
                "worker",
                "start",
                "-p",
                "test",
                "--run-once",
                "-t",
                "process",
            ],
            expected_code=0,
        )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name="test",
        work_queues=None,
        prefetch_seconds=100,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_with_limit(mock_worker):
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-l",
            "5",
            "-p",
            "test",
            "--run-once",
            "-t",
            "process",
        ],
        expected_code=0,
    )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name="test",
        work_queues=None,
        prefetch_seconds=10,
        limit=5,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_create_pool_if_not_found_default(mock_worker):
    """Omitting the flag passes create_pool_if_not_found=True (the default)."""
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-p",
            "test",
            "--run-once",
            "-t",
            "process",
        ],
        expected_code=0,
    )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name="test",
        work_queues=None,
        prefetch_seconds=ANY,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=True,
    )


@pytest.mark.usefixtures("use_hosted_api_server")
def test_start_worker_no_create_pool_if_not_found(mock_worker):
    """--no-create-pool-if-not-found passes create_pool_if_not_found=False to the worker."""
    invoke_and_assert(
        command=[
            "worker",
            "start",
            "-p",
            "test",
            "--run-once",
            "-t",
            "process",
            "--no-create-pool-if-not-found",
        ],
        expected_code=0,
    )
    mock_worker.assert_called_once_with(
        name=None,
        work_pool_name="test",
        work_queues=None,
        prefetch_seconds=ANY,
        limit=None,
        heartbeat_interval_seconds=30,
        base_job_template=None,
        create_pool_if_not_found=False,
    )
    mock_worker.return_value.start.assert_awaited_once_with(
        run_once=True, with_healthcheck=False, printer=ANY
    )


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_worker_joins_existing_pool(work_pool, prefect_client: PrefectClient):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            work_pool.name,
            "-n",
            "test-worker",
            "-t",
            "process",
        ],
        expected_code=0,
        expected_output_contains=[
            "Worker 'test-worker' started!",
            "Worker 'test-worker' stopped!",
        ],
    )

    workers = await prefect_client.read_workers_for_work_pool(
        work_pool_name=work_pool.name
    )
    assert workers[0].name == "test-worker"


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_worker_discovers_work_pool_type(
    process_work_pool, prefect_client: PrefectClient
):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            process_work_pool.name,
            "-n",
            "test-worker",
        ],
        expected_code=0,
        expected_output_contains=[
            (
                f"Discovered type {process_work_pool.type!r} for work pool"
                f" {process_work_pool.name!r}."
            ),
            "Worker 'test-worker' started!",
            "Worker 'test-worker' stopped!",
        ],
    )

    workers = await prefect_client.read_workers_for_work_pool(
        work_pool_name=process_work_pool.name
    )
    assert workers[0].name == "test-worker"


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_worker_start_fails_informatively_with_bad_type(
    process_work_pool, prefect_client: PrefectClient
):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "-p",
            process_work_pool.name,
            "-t",
            "not-a-real-type",
        ],
        expected_code=1,
        expected_output_contains=[
            "Could not find a package for worker type",
            "Unable to start worker. Please ensure you have the necessary"
            " dependencies installed to run your desired worker type.",
        ],
    )


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_worker_does_not_run_with_push_pool(push_work_pool):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            push_work_pool.name,
        ],
        expected_code=1,
        expected_output_contains=[
            (
                f"Discovered type {push_work_pool.type!r} for work pool"
                f" {push_work_pool.name!r}."
            ),
            (
                "Workers are not required for push work pools. "
                "See https://docs.prefect.io/latest/deploy/infrastructure-examples/serverless "
                "for more details."
            ),
        ],
    )


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_start_worker_without_type_creates_process_work_pool(
    prefect_client: PrefectClient,
):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            "not-here",
            "-n",
            "test-worker",
        ],
        expected_code=0,
        expected_output_contains=[
            (
                "Work pool 'not-here' does not exist and no worker type was"
                " provided. Starting a process worker..."
            ),
            "Worker 'test-worker' started!",
            "Worker 'test-worker' stopped!",
        ],
    )

    workers = await prefect_client.read_workers_for_work_pool(work_pool_name="not-here")
    assert workers[0].name == "test-worker"


@pytest.mark.usefixtures("use_hosted_api_server")
async def test_worker_reports_heartbeat_interval(
    prefect_client: PrefectClient, process_work_pool
):
    await run_sync_in_worker_thread(
        invoke_and_assert,
        command=[
            "worker",
            "start",
            "--run-once",
            "-p",
            process_work_pool.name,
            "-n",
            "test-worker",
        ],
        expected_code=0,
        expected_output_contains=[
            "Worker 'test-worker' started!",
            "Worker 'test-worker' stopped!",
        ],
    )

    workers = await prefect_client.read_workers_for_work_pool(
        work_pool_name=process_work_pool.name
    )
    assert len(workers) == 1
    assert workers[0].name == "test-worker"
    assert workers[0].heartbeat_interval_seconds == 30


@pytest.mark.usefixtures("use_hosted_api_server")
class TestInstallPolicyOption:
    async def test_install_policy_if_not_present(
        self, kubernetes_work_pool, monkeypatch
    ):
        import prefect.cli._worker_utils

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.side_effect = [KeyError, MockKubernetesWorker]
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
                "--install-policy=if-not-present",
            ],
            expected_output_contains=[
                "Installing prefect-kubernetes...",
                "Worker 'test-worker' started!",
                "Worker 'test-worker' stopped!",
            ],
        )

        run_process_mock.assert_called_once_with(
            [uv.find_uv_bin(), "pip", "install", "prefect[kubernetes]"],
            stream_output=True,
        )

    @pytest.mark.usefixtures("interactive_console")
    async def test_install_policy_prompt(self, kubernetes_work_pool, monkeypatch):
        import prefect.cli._worker_utils

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.side_effect = [KeyError, MockKubernetesWorker]
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
            ],
            user_input=readchar.key.ENTER,
            expected_output_contains=[
                "Could not find the Prefect integration library for the",
                "kubernetes",
                "Install the library now?",
                "Installing prefect-kubernetes...",
                "Worker 'test-worker' started!",
                "Worker 'test-worker' stopped!",
            ],
        )

        run_process_mock.assert_called_once_with(
            [uv.find_uv_bin(), "pip", "install", "prefect[kubernetes]"],
            stream_output=True,
        )

    @pytest.mark.usefixtures("interactive_console")
    async def test_install_policy_prompt_decline(self, monkeypatch, prefect_client):
        import prefect.cli._worker_utils

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.side_effect = [KeyError, MockKubernetesWorker]
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        kubernetes_work_pool = await prefect_client.create_work_pool(
            work_pool=WorkPoolCreate(name="test-k8s-work-pool", type="kubernetes")
        )

        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
            ],
            expected_code=1,
            user_input="n" + readchar.key.ENTER,
            expected_output_contains=[
                "Unable to start worker. Please ensure you have the necessary"
                " dependencies installed to run your desired worker type."
            ],
        )

        run_process_mock.assert_not_called()

    @pytest.mark.usefixtures("interactive_console")
    async def test_install_policy_if_not_present_overrides_prompt(
        self, kubernetes_work_pool, monkeypatch
    ):
        import prefect.cli._worker_utils

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.side_effect = [KeyError, MockKubernetesWorker]
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
                "--install-policy=if-not-present",
            ],
            expected_output_contains=[
                "Installing prefect-kubernetes...",
                "Worker 'test-worker' started!",
                "Worker 'test-worker' stopped!",
            ],
        )

        run_process_mock.assert_called_once_with(
            [uv.find_uv_bin(), "pip", "install", "prefect[kubernetes]"],
            stream_output=True,
        )

    @pytest.mark.usefixtures("interactive_console")
    async def test_install_policy_always(self, kubernetes_work_pool, monkeypatch):
        import prefect.cli._worker_utils

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.return_value = MockKubernetesWorker
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
                "--install-policy=always",
            ],
            expected_output_contains=[
                "Installing prefect-kubernetes...",
                "Worker 'test-worker' started!",
                "Worker 'test-worker' stopped!",
            ],
        )

        run_process_mock.assert_called_once_with(
            [uv.find_uv_bin(), "pip", "install", "prefect[kubernetes]", "--upgrade"],
            stream_output=True,
        )

    @pytest.mark.usefixtures("interactive_console")
    async def test_install_policy_never(self, monkeypatch, prefect_client):
        import prefect.cli._worker_utils

        kubernetes_work_pool = await prefect_client.create_work_pool(
            work_pool=WorkPoolCreate(name="test-k8s-work-pool", type="kubernetes")
        )

        run_process_mock = AsyncMock()
        lookup_type_mock = MagicMock()
        lookup_type_mock.side_effect = KeyError
        monkeypatch.setattr(
            "prefect.utilities.processutils.run_process", run_process_mock
        )
        monkeypatch.setattr(prefect.cli._worker_utils, "lookup_type", lookup_type_mock)
        await run_sync_in_worker_thread(
            invoke_and_assert,
            command=[
                "worker",
                "start",
                "--run-once",
                "-p",
                kubernetes_work_pool.name,
                "-n",
                "test-worker",
                "--install-policy=never",
            ],
            expected_code=1,
            expected_output_contains=[
                "Unable to start worker. Please ensure you have the necessary"
                " dependencies installed to run your desired worker type."
            ],
        )

        run_process_mock.assert_not_called()

        def test_start_with_prefect_agent_type(worker_type):
            invoke_and_assert(
                command=[
                    "worker",
                    "start",
                    "--run-once",
                    "-p",
                    "test-work-pool",
                    "-n",
                    "test-worker",
                    "-t",
                    "prefect-agent",
                ],
                expected_code=1,
                expected_output_contains=(
                    "'prefect-agent' typed work pools work with Prefect Agents instead"
                    " of Workers. Please use the 'prefect agent start' to start a"
                    " Prefect Agent."
                ),
            )


POLL_INTERVAL = 0.5
STARTUP_TIMEOUT = 20
SHUTDOWN_TIMEOUT = 5


async def safe_shutdown(process):
    try:
        with anyio.fail_after(SHUTDOWN_TIMEOUT):
            await process.wait()
    except TimeoutError:
        # try twice in case process.wait() hangs
        with anyio.fail_after(SHUTDOWN_TIMEOUT):
            await process.wait()


@pytest.fixture(scope="function")
async def worker_process(use_hosted_api_server):
    """
    Runs an agent listening to all queues.
    Yields:
        The anyio.Process.
    """
    out = tempfile.TemporaryFile()  # capture output for test assertions

    # Will connect to the same database as normal test clients
    async with open_process(
        command=[
            "prefect",
            "worker",
            "start",
            "--type",
            "process",
            "--pool",
            "my-pool",
            "--name",
            "test-worker",
        ],
        stdout=out,
        stderr=out,
        env={**os.environ, **get_current_settings().to_environment_variables()},
    ) as process:
        process.out = out

        for _ in range(int(STARTUP_TIMEOUT / POLL_INTERVAL)):
            await anyio.sleep(POLL_INTERVAL)
            if out.tell() > 400:
                # Sleep to allow startup to complete
                # TODO: Replace with a healthcheck endpoint
                await anyio.sleep(4)
                break

        assert out.tell() > 400, "The worker did not start up in time"
        assert process.returncode is None, "The worker failed to start up"

        # Yield to the consuming tests
        yield process

        # Then shutdown the process
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        out.close()


@asynccontextmanager
async def worker_with_owned_infrastructure(
    tmp_path: Path, **kwargs: object
) -> AsyncGenerator[anyio.abc.Process, None]:
    """Own launched supervisor/engine processes before they import flow code."""
    launcher = tmp_path / "owned_launcher.py"
    launcher.write_text(
        dedent(
            """
            import array
            import os
            import socket
            import sys

            # Register before executing any supervisor/preparation/engine code.
            with socket.socket(fileno=int(os.environ["DRAIN_TEST_OWNER_FD"])) as owner:
                handle = os.pidfd_open(os.getpid())
                try:
                    owner.sendmsg(
                        [str(os.getpid()).encode()],
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [handle]))],
                    )
                finally:
                    os.close(handle)
            os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
            """
        )
    )
    bootstrap = tmp_path / "owned_worker.py"
    bootstrap.write_text(
        dedent(
            f"""
            import os
            import runpy
            import sys

            import anyio

            original_open = anyio.open_process
            owner_fd = int(os.environ["DRAIN_TEST_OWNER_FD"])

            async def owned_open(command, *args, **kwargs):
                kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), owner_fd)
                return await original_open(
                    [sys.executable, {str(launcher)!r}, *command], *args, **kwargs
                )

            anyio.open_process = owned_open
            sys.argv = sys.argv[1:]
            runpy.run_path(sys.argv[0], run_name="__main__")
            """
        )
    )
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with reader, writer:
        command = kwargs.pop("command")
        kwargs["env"] = {
            **kwargs["env"],
            "DRAIN_TEST_OWNER_FD": str(writer.fileno()),
        }
        kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), writer.fileno())
        process = await anyio.open_process(
            [sys.executable, str(bootstrap), *command[1:]], **kwargs
        )
        writer.close()
        owned = []
        try:
            yield process
        finally:
            with anyio.CancelScope(shield=True), anyio.fail_after(10):
                # Stop the spawning authority first, then drain ownership transfers.
                # A launcher in startup still holds the socket: EOF cannot precede
                # its registration, even if the worker dies before it sends.
                if process.returncode is None:
                    process.kill()
                await process.aclose()
                while True:
                    await anyio.wait_readable(reader)
                    handles = array.array("i")
                    message, ancillary, flags, _ = reader.recvmsg(
                        64, socket.CMSG_SPACE(handles.itemsize), socket.MSG_DONTWAIT
                    )
                    if not message:
                        break
                    for level, kind, data in ancillary:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            handles.frombytes(data)
                    assert len(handles) == 1 and not flags
                    handle = handles[0]
                    try:
                        was_alive = not select.select([handle], [], [], 0)[0]
                        try:
                            signal.pidfd_send_signal(handle, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await anyio.wait_readable(handle)
                        owned.append(
                            {
                                "pid": int(message),
                                "alive_before_cleanup": bool(was_alive),
                                "exited": bool(select.select([handle], [], [], 0)[0]),
                            }
                        )
                    finally:
                        os.close(handle)
                (tmp_path / "cleanup.json").write_text(
                    json.dumps(
                        {
                            "owned": owned,
                            "worker_exited": process.returncode is not None,
                        }
                    )
                )


class TestWorkerSignalForwarding:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="SIGTERM is only used in non-Windows environments",
    )
    async def test_sigint_drains_worker(self, worker_process):
        worker_process.send_signal(signal.SIGINT)
        await safe_shutdown(worker_process)
        worker_process.out.seek(0)
        out = worker_process.out.read().decode()

        assert "Draining the process worker" in out, (
            "When sending a SIGINT, the worker should drain active runs."
            f" Output:\n{out}"
        )
        assert "Worker 'test-worker' stopped!" in out, (
            "When sending a SIGINT, the main process should shutdown gracefully."
            f" Output:\n{out}"
        )

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="SIGTERM is only used in non-Windows environments",
    )
    async def test_sigterm_drains_process_worker(self, worker_process):
        worker_process.send_signal(signal.SIGTERM)
        await safe_shutdown(worker_process)
        worker_process.out.seek(0)
        out = worker_process.out.read().decode()

        assert "Draining the process worker" in out, (
            "When sending a SIGTERM, the process worker should drain active runs."
            f" Output:\n{out}"
        )
        assert "Worker 'test-worker' stopped!" in out, (
            "When sending a SIGTERM, the main process should shutdown gracefully."
            f" Output:\n{out}"
        )

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "limit,signum",
        [
            (None, signal.SIGTERM),
            (None, signal.SIGINT),
            (1, signal.SIGTERM),
            (1, signal.SIGINT),
            pytest.param(None, None, id="failure-before-handshake"),
        ],
    )
    @pytest.mark.skipif(
        not hasattr(os, "pidfd_open"),
        reason="Linux pidfds are required to verify actual child exit",
    )
    async def test_signal_waits_for_real_process_exit_before_worker_stops(
        self,
        signum: int | None,
        limit: int | None,
        use_hosted_api_server: str,
        prefect_client: PrefectClient,
        tmp_path: Path,
    ):
        fail_before_handshake = signum is None
        worker_name = f"drain-e2e-{uuid.uuid4().hex}"
        work_pool_name = f"drain-e2e-{uuid.uuid4().hex}"
        await prefect_client.create_work_pool(
            WorkPoolCreate(name=work_pool_name, type="process")
        )

        listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
        _, port = listener.extra(SocketAttribute.local_address)
        pid_file = tmp_path / "child.pid"
        flow_file = tmp_path / "drain_flow.py"
        flow_file.write_text(
            "import atexit\n"
            "import os\n"
            "import socket\n\n"
            "from prefect import flow\n\n"
            "connection = None\n\n"
            "def wait_before_process_exit():\n"
            "    connection.sendall(b'E')\n"
            "    connection.recv(1)\n"
            "    connection.close()\n\n"
            "@flow\n"
            "def wait_for_release():\n"
            "    global connection\n"
            "    connection = socket.create_connection(\n"
            "        ('127.0.0.1', int(os.environ['DRAIN_TEST_PORT']))\n"
            "    )\n"
            f"    with open({str(pid_file)!r}, 'w') as pid_file:\n"
            "        pid_file.write(str(os.getpid()))\n"
            "    connection.sendall(b'S')\n"
            "    connection.recv(1)\n"
            "    atexit.register(wait_before_process_exit)\n"
        )

        if fail_before_handshake:
            flow_file.write_text(
                "import os\nimport socket\nimport threading\n"
                f"with open({str(pid_file)!r}, 'w') as pid_file:\n"
                "    pid_file.write(str(os.getpid()))\n"
                f"startup = socket.create_connection(('127.0.0.1', {port}))\n"
                "startup.sendall(b'I')\n"
                "threading.Event().wait()\n" + flow_file.read_text()
            )

        source_observation = tmp_path / "stopped-emission.json"
        bootstrap = tmp_path / "worker_cli.py"
        bootstrap.write_text(
            dedent(
                f"""
                import array
                import json
                import os
                import select
                import socket
                from pathlib import Path

                from prefect.cli import app
                from prefect.workers.base import BaseWorker

                original_emit = BaseWorker._emit_worker_stopped_event

                async def observe_stopped_emission(self, started_event):
                    handles = array.array("i")
                    with socket.socket(
                        fileno=int(os.environ["DRAIN_TEST_OBSERVER_FD"])
                    ) as observer:
                        message, ancillary, flags, _ = observer.recvmsg(
                            64, socket.CMSG_SPACE(handles.itemsize),
                            socket.MSG_DONTWAIT,
                        )
                    for level, kind, data in ancillary:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            handles.frombytes(data)
                    assert len(handles) == 1 and not flags
                    try:
                        child_exited = bool(select.select(handles, [], [], 0)[0])
                        Path({str(source_observation)!r}).write_text(json.dumps({{
                            "worker_pid": os.getpid(),
                            "child_pid": int(message),
                            "child_exited": child_exited,
                        }}))
                    finally:
                        os.close(handles[0])
                    await original_emit(self, started_event)

                BaseWorker._emit_worker_stopped_event = observe_stopped_emission
                app()
                """
            )
        )

        @flow
        def drain_test_flow():
            pass

        flow_id = await prefect_client.create_flow(drain_test_flow)
        deployment_id = await prefect_client.create_deployment(
            flow_id=flow_id,
            name=f"drain-e2e-{uuid.uuid4().hex}",
            work_pool_name=work_pool_name,
            path=str(tmp_path),
            entrypoint=f"{flow_file.name}:wait_for_release",
            job_variables={"env": {"DRAIN_TEST_PORT": str(port)}},
        )
        flow_run = await prefect_client.create_flow_run_from_deployment(
            deployment_id=deployment_id,
        )

        import_started = anyio.Event()
        child_started = anyio.Event()
        flow_application_completed = anyio.Event()
        release_flow = anyio.Event()
        release_process_exit = anyio.Event()
        worker_started = anyio.Event()
        drain_started = anyio.Event()
        flow_run_completed = anyio.Event()
        worker_stopped = anyio.Event()
        cli_stopped = anyio.Event()
        cli_exited = anyio.Event()
        child_pidfd: int | None = None
        observer_socket: socket.socket | None = None
        descriptors = ExitStack()
        output = ""

        async def control_child(stream: SocketStream) -> None:
            nonlocal child_pidfd
            async with stream:
                marker = await stream.receive(1)
                if fail_before_handshake:
                    assert marker == b"I"
                    import_started.set()
                    await anyio.sleep_forever()
                assert marker == b"S"
                child_pidfd = os.pidfd_open(int(pid_file.read_text()))
                descriptors.callback(os.close, child_pidfd)
                # Transfer the exact flow child's open pidfd to the worker before
                # signalling it. No PID lookup after exit or supervisor PID is used.
                assert observer_socket is not None
                observer_socket.sendmsg(
                    [pid_file.read_bytes()],
                    [
                        (
                            socket.SOL_SOCKET,
                            socket.SCM_RIGHTS,
                            array.array("i", [child_pidfd]),
                        )
                    ],
                )
                child_started.set()
                await release_flow.wait()
                await stream.send(b"R")
                assert await stream.receive(1) == b"E"
                flow_application_completed.set()
                await release_process_exit.wait()
                await stream.send(b"X")
                with pytest.raises(anyio.EndOfStream):
                    await stream.receive(1)

        def assert_child_exited() -> None:
            # Socket EOF and the Completed state both precede OS process exit.
            # A readable pidfd proves that this exact child has actually exited.
            assert child_pidfd is not None
            assert select.select([child_pidfd], [], [], 0)[0], output

        def assert_source_observation() -> None:
            assert json.loads(source_observation.read_text()) == {
                "worker_pid": process.pid,
                "child_pid": int(pid_file.read_text()),
                "child_exited": True,
            }

        @asynccontextmanager
        async def worker_process(
            **kwargs: object,
        ) -> AsyncGenerator[anyio.abc.Process, None]:
            nonlocal observer_socket
            observer_socket, worker_socket = socket.socketpair()
            with observer_socket, worker_socket:
                kwargs["env"] = {
                    **kwargs["env"],
                    "DRAIN_TEST_OBSERVER_FD": str(worker_socket.fileno()),
                }
                async with worker_with_owned_infrastructure(
                    tmp_path,
                    start_new_session=True,
                    pass_fds=[worker_socket.fileno()],
                    **kwargs,
                ) as process:
                    worker_socket.close()
                    try:
                        yield process
                    finally:
                        release_flow.set()
                        release_process_exit.set()

        worker_event_filter = EventFilter(
            event=EventNameFilter(name=["prefect.worker.stopped"]),
            resource=EventResourceFilter(id=[f"prefect.worker.process.{worker_name}"]),
        )
        flow_event_filter = EventFilter(
            event=EventNameFilter(name=["prefect.flow-run.Completed"]),
            resource=EventResourceFilter(id=[f"prefect.flow-run.{flow_run.id}"]),
        )

        failure_context = (
            pytest.raises(ExceptionGroup) if fail_before_handshake else nullcontext()
        )
        with descriptors, failure_context as failure:
            async with (
                listener,
                PrefectEventSubscriber(
                    api_url=use_hosted_api_server,
                    filter=worker_event_filter,
                ) as worker_subscriber,
                PrefectEventSubscriber(
                    api_url=use_hosted_api_server,
                    filter=flow_event_filter,
                ) as flow_subscriber,
                worker_process(
                    command=[
                        sys.executable,
                        str(bootstrap),
                        "worker",
                        "start",
                        "--type",
                        "process",
                        "--pool",
                        work_pool_name,
                        "--name",
                        worker_name,
                        *(["--limit", str(limit)] if limit is not None else []),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        **get_current_settings().to_environment_variables(),
                    },
                ) as process,
                anyio.create_task_group() as task_group,
            ):

                async def read_output() -> None:
                    nonlocal output
                    assert process.stdout is not None
                    async for chunk in process.stdout:
                        output += chunk.decode(errors="replace")
                        if f"Worker '{worker_name}' started!" in output:
                            worker_started.set()
                        if "Draining the process worker" in output:
                            drain_started.set()
                        if f"Worker '{worker_name}' stopped!" in output:
                            assert_child_exited()
                            cli_stopped.set()

                async def read_flow_run_completed_event() -> None:
                    async for _ in flow_subscriber:
                        flow_run_completed.set()
                        return

                async def read_worker_stopped_event() -> None:
                    async for _ in worker_subscriber:
                        assert_source_observation()
                        assert_child_exited()
                        worker_stopped.set()
                        return

                async def wait_for_cli_exit() -> None:
                    await process.wait()
                    assert_child_exited()
                    cli_exited.set()

                task_group.start_soon(listener.serve, control_child)
                task_group.start_soon(read_output)
                task_group.start_soon(read_flow_run_completed_event)
                task_group.start_soon(read_worker_stopped_event)
                task_group.start_soon(wait_for_cli_exit)

                with anyio.fail_after(20):
                    await worker_started.wait()
                    if fail_before_handshake:
                        await import_started.wait()
                        assert child_pidfd is None
                        assert not child_started.is_set()
                        raise RuntimeError("injected failure before S")
                    await child_started.wait()

                process.send_signal(signum)
                with anyio.fail_after(10):
                    await drain_started.wait()

                assert process.returncode is None
                assert not flow_application_completed.is_set()
                assert not worker_stopped.is_set()
                assert not cli_stopped.is_set()
                assert not cli_exited.is_set()

                release_flow.set()
                with anyio.fail_after(20):
                    await flow_application_completed.wait()
                    await flow_run_completed.wait()

                completed_flow_run = await prefect_client.read_flow_run(flow_run.id)
                assert completed_flow_run.state is not None
                assert completed_flow_run.state.is_completed()
                assert process.returncode is None
                assert not worker_stopped.is_set()
                assert not cli_stopped.is_set()
                assert not cli_exited.is_set()

                release_process_exit.set()
                with anyio.fail_after(20):
                    await worker_stopped.wait()
                    await cli_stopped.wait()
                    await cli_exited.wait()

                assert_source_observation()
                assert_child_exited()
                assert process.returncode == 0, output
                task_group.cancel_scope.cancel()

        if fail_before_handshake:
            assert failure is not None
            assert len(failure.value.exceptions) == 1
            error = failure.value.exceptions[0]
            assert isinstance(error, RuntimeError)
            assert str(error) == "injected failure before S"
            cleanup = json.loads((tmp_path / "cleanup.json").read_text())
            assert cleanup["worker_exited"]
            assert cleanup["owned"]
            assert all(process["exited"] for process in cleanup["owned"])
            assert any(
                process["pid"] == int(pid_file.read_text())
                and process["alive_before_cleanup"]
                for process in cleanup["owned"]
            )

    async def test_sigint_sends_sigterm_then_sigkill(self, worker_process):
        worker_process.send_signal(signal.SIGINT)
        await anyio.sleep(0.1)  # some time needed for the recursive signal handler
        worker_process.send_signal(signal.SIGINT)
        await safe_shutdown(worker_process)
        worker_process.out.seek(0)
        out = worker_process.out.read().decode()

        if sys.platform != "win32":
            assert (
                # either the main PID is still waiting for shutdown, so forwards the SIGKILL
                "Sending SIGKILL" in out
                # or SIGKILL came too late, and the main PID is already closing
                or "KeyboardInterrupt" in out
                or "Worker 'test-worker' stopped!" in out
                or "Aborted." in out
            ), (
                "When sending two SIGINT shortly after each other, the main process should"
                f" first receive a SIGINT and then a SIGKILL. Output:\n{out}"
            )
        else:
            assert "Sending CTRL_BREAK_EVENT" in out, (
                "When sending a SIGINT, the main process should send a CTRL_BREAK_EVENT to"
                f" the worker subprocess. Output:\n{out}"
            )

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="SIGTERM is only used in non-Windows environments",
    )
    async def test_sigterm_sends_sigterm_then_sigkill(self, worker_process):
        worker_process.send_signal(signal.SIGTERM)
        await anyio.sleep(0.1)  # some time needed for the recursive signal handler
        worker_process.send_signal(signal.SIGTERM)
        await safe_shutdown(worker_process)
        worker_process.out.seek(0)
        out = worker_process.out.read().decode()

        assert (
            # either the main PID is still waiting for shutdown, so forwards the SIGKILL
            "Sending SIGKILL" in out
            # or SIGKILL came too late, and the main PID is already closing
            or "KeyboardInterrupt" in out
            or "Worker 'test-worker' stopped!" in out
            or "Aborted." in out
        ), (
            "When sending two SIGTERM shortly after each other, the main process should"
            f" drain and then receive a SIGKILL. Output:\n{out}"
        )
