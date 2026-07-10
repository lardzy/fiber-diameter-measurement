from __future__ import annotations

import unittest

from app.security import AuthConfigurationError, AuthenticationError, auth_status, authenticate


class SecurityTests(unittest.TestCase):
    def test_auth_is_fail_closed_without_token_or_explicit_dev_mode(self) -> None:
        environment: dict[str, str] = {}

        self.assertEqual(auth_status(environment), (False, "api_auth_not_configured"))
        with self.assertRaises(AuthConfigurationError):
            authenticate(None, environment)

    def test_anonymous_access_requires_explicit_dev_switch(self) -> None:
        environment = {"AREA_ALLOW_ANONYMOUS_DEV": "true"}

        self.assertEqual(auth_status(environment), (True, "anonymous-dev"))
        self.assertEqual(authenticate(None, environment), "anonymous-dev")

    def test_bearer_token_is_compared_and_short_tokens_are_rejected(self) -> None:
        short = {"AREA_API_TOKEN": "too-short"}
        self.assertEqual(auth_status(short), (False, "api_token_too_short"))

        environment = {"AREA_API_TOKEN": "0123456789abcdef0123456789abcdef"}
        self.assertEqual(authenticate("Bearer 0123456789abcdef0123456789abcdef", environment), "token")
        with self.assertRaises(AuthenticationError):
            authenticate("Bearer wrong-token-value", environment)


if __name__ == "__main__":
    unittest.main()
