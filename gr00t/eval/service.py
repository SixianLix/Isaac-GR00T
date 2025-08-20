# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Optional
import numpy as np

import torch
# import zmq

import asyncio
import msgpack_numpy as m
import websockets

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555, api_token: str = None):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(TorchSerializer.to_bytes({"error": str(e)}))


# class BaseInferenceClient:
#     def __init__(
#         self,
#         host: str = "localhost",
#         port: int = 5555,
#         timeout_ms: int = 15000,
#         api_token: str = None,
#     ):
#         self.context = zmq.Context()
#         self.host = host
#         self.port = port
#         self.timeout_ms = timeout_ms
#         self.api_token = api_token
#         self._init_socket()

#     def _init_socket(self):
#         """Initialize or reinitialize the socket with current settings"""
#         self.socket = self.context.socket(zmq.REQ)
#         self.socket.connect(f"tcp://{self.host}:{self.port}")

#     def ping(self) -> bool:
#         try:
#             self.call_endpoint("ping", requires_input=False)
#             return True
#         except zmq.error.ZMQError:
#             self._init_socket()  # Recreate socket for next attempt
#             return False

#     def kill_server(self):
#         """
#         Kill the server.
#         """
#         self.call_endpoint("kill", requires_input=False)

#     def call_endpoint(
#         self, endpoint: str, data: dict | None = None, requires_input: bool = True
#     ) -> dict:
#         """
#         Call an endpoint on the server.

#         Args:
#             endpoint: The name of the endpoint.
#             data: The input data for the endpoint.
#             requires_input: Whether the endpoint requires input data.
#         """
#         import ipdb;ipdb.set_trace()
#         request: dict = {"endpoint": endpoint}
#         if requires_input:
#             request["data"] = data
#         if self.api_token:
#             request["api_token"] = self.api_token

#         self.socket.send(TorchSerializer.to_bytes(request))
#         message = self.socket.recv()
#         response = TorchSerializer.from_bytes(message)

#         if "error" in response:
#             raise RuntimeError(f"Server error: {response['error']}")
#         return response

#     def __del__(self):
#         """Cleanup resources on destruction"""
#         self.socket.close()
#         self.context.term()



class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout_ms: int = 150_000,
        api_token: Optional[str] = None,
    ):
        self.url = f"ws://{host}:{port}"
        self.timeout = timeout_ms / 1000  # 秒
        self.api_token = api_token

    # ---------- 公共同步外观（内部转异步） ----------
    def ping(self) -> bool:
        return asyncio.run(self._ping_async())

    def kill_server(self) -> None:
        return asyncio.run(self._kill_async())

    def call_endpoint(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        requires_input: bool = True,
    ) -> Dict[str, Any]:
        """同步外壳，方便旧代码调用"""
        return asyncio.run(self._call_async(endpoint, data, requires_input))

    # ---------- 异步实现 ----------
    async def _ping_async(self) -> bool:
        try:
            await self._call_async("ping", requires_input=False)
            return True
        except Exception as e:
            print(f"Ping failed: {e}")
            return False

    async def _kill_async(self) -> None:
        await self._call_async("kill", requires_input=False)

    async def _call_async(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        requires_input: bool = True,
    ) -> Dict[str, Any]:
        async with websockets.connect(self.url, close_timeout=self.timeout) as ws:
            await ws.send(m.packb(data))
            raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            response = m.unpackb(raw)

            def try_decode_numpy(val):
                # 兼容两种格式:
                # 1. {b'nd': True, b'type': ..., b'shape': ..., b'data': ...}
                # 2. {b'__ndarray__': True, b'data': ..., b'dtype': ..., b'shape': ...}
                if isinstance(val, dict):
                    # 格式1
                    if (
                        b'nd' in val
                        and b'type' in val
                        and b'shape' in val
                        and b'data' in val
                    ):
                        dtype = np.dtype(val[b'type'].decode() if isinstance(val[b'type'], bytes) else val[b'type'])
                        arr = np.frombuffer(val[b'data'], dtype=dtype)
                        arr = arr.reshape(val[b'shape'])
                        return arr
                    # 格式2
                    if (
                        b'__ndarray__' in val
                        and b'data' in val
                        and b'dtype' in val
                        and b'shape' in val
                    ):
                        dtype = np.dtype(val[b'dtype'].decode() if isinstance(val[b'dtype'], bytes) else val[b'dtype'])
                        arr = np.frombuffer(val[b'data'], dtype=dtype)
                        arr = arr.reshape(val[b'shape'])
                        return arr
                return val

            # 递归处理嵌套字典
            def decode_all(obj):
                if isinstance(obj, dict):
                    return {k.decode() if isinstance(k, bytes) else k: decode_all(try_decode_numpy(v)) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [decode_all(v) for v in obj]
                else:
                    return obj

            response = decode_all(response)
            if "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            return response


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)

if __name__ == "__main__":
    client = BaseInferenceClient()
    print("ping:", client.ping())
    # 显式手动关
    client.socket.close()