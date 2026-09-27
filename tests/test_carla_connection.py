import unittest


class CarlaConnectionTests(unittest.TestCase):
    def _resolve(self, environ):
        try:
            from vladfuzz_runtime.carla_connection import resolve_carla_connection
        except ModuleNotFoundError as exc:
            self.fail(f"CARLA connection resolver is missing: {exc}")
        return resolve_carla_connection(environ)

    def test_default_execution_keeps_existing_connection_defaults(self):
        connection = self._resolve(
            {
                "CARLA_HOST": "127.0.0.2",
                "CARLA_PORT": "2010",
                "CARLA_TM_PORT": "8010",
            }
        )

        self.assertEqual(connection.host, "localhost")
        self.assertEqual(connection.rpc_port, 2000)
        self.assertIsNone(connection.tm_port)

    def test_isolated_execution_uses_configured_connection(self):
        connection = self._resolve(
            {
                "CARLA_ISOLATED": "1",
                "CARLA_HOST": "127.0.0.1",
                "CARLA_PORT": "2010",
                "CARLA_TM_PORT": "8010",
            }
        )

        self.assertEqual(connection.host, "127.0.0.1")
        self.assertEqual(connection.rpc_port, 2010)
        self.assertEqual(connection.tm_port, 8010)

    def test_isolated_execution_rejects_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "CARLA_PORT"):
            self._resolve(
                {
                    "CARLA_ISOLATED": "1",
                    "CARLA_PORT": "invalid",
                    "CARLA_TM_PORT": "8010",
                }
            )


if __name__ == "__main__":
    unittest.main()
