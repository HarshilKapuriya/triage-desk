"""Focused unit tests for the disposable target safety validator.

Verifies:
1. Valid authorized target acceptance (strict 127.0.0.1:5433/triage_events_disposable/test_user).
2. Input sanitization (None, empty, whitespace, malformed URLs).
3. Host validation (rejection of localhost, external IPs, domain names, multiple hosts).
4. Hostaddr validation (rejection of overrides, mismatched IPs, multiple hostaddrs).
5. Port validation (rejection of 5432, missing port, multiple ports).
6. Database name validation (rejection of test_triage_desk, postgres, missing dbname).
7. User validation (rejection of postgres, root, missing user).
8. Service and servicefile rejection in conninfo.
9. Ambient libpq environment variable inspection (PGSERVICE, PGSERVICEFILE, PGSYSCONFDIR,
   PGHOST, PGHOSTADDR, PGPORT, PGDATABASE, PGUSER).
10. Unexpected connection parameter whitelist enforcement.
11. Cloud infrastructure keyword rejection (supabase, pooler, aws, neon, rds, azure, gcp).
12. Password and secret confidentiality: error messages and exceptions must never leak credentials.
"""

import pytest
from tests.target_validator import (
    validate_isolated_target,
    TargetSafetyError,
    REQUIRED_TARGET,
)

VALID_DISPOSABLE_URL = (
    "postgresql://test_user:super_secret_test_pw@127.0.0.1:5433/triage_events_disposable"
)


# 1. Acceptance of authorized disposable target
def test_valid_disposable_target_accepted():
    params = validate_isolated_target(VALID_DISPOSABLE_URL, env={})
    assert params["host"] == "127.0.0.1"
    assert params["port"] == "5433"
    assert params["dbname"] == "triage_events_disposable"
    assert params["user"] == "test_user"


def test_valid_disposable_target_with_allowed_params():
    url = f"{VALID_DISPOSABLE_URL}?sslmode=disable&application_name=unit_test"
    params = validate_isolated_target(url, env={})
    assert params["sslmode"] == "disable"
    assert params["application_name"] == "unit_test"


def test_valid_disposable_target_with_matching_hostaddr():
    url = f"{VALID_DISPOSABLE_URL}?hostaddr=127.0.0.1"
    params = validate_isolated_target(url, env={})
    assert params["hostaddr"] == "127.0.0.1"


# 2. Input sanitization
@pytest.mark.parametrize("empty_input", [None, "", "   ", "\t\n"])
def test_reject_empty_or_none(empty_input):
    with pytest.raises(TargetSafetyError, match="empty or None"):
        validate_isolated_target(empty_input, env={})


def test_reject_malformed_url():
    with pytest.raises(TargetSafetyError, match="Malformed connection URL"):
        validate_isolated_target(":::not_a_valid_url:::", env={})


# 3. Host validation
def test_reject_localhost():
    url = "postgresql://test_user:test_password@localhost:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Host must be exactly '127.0.0.1'"):
        validate_isolated_target(url, env={})


def test_reject_external_host():
    url = "postgresql://test_user:test_password@192.168.1.10:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Host must be exactly '127.0.0.1'"):
        validate_isolated_target(url, env={})


def test_reject_missing_host():
    url = "postgresql://test_user:test_password@:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Missing required host parameter"):
        validate_isolated_target(url, env={})


def test_reject_multiple_hosts():
    url = "postgresql://test_user:test_password@127.0.0.1,192.168.1.2:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Multiple host or port targets are rejected"):
        validate_isolated_target(url, env={})


# 4. Hostaddr validation
def test_reject_alternate_hostaddr_override():
    url = f"{VALID_DISPOSABLE_URL}?hostaddr=10.0.0.1"
    with pytest.raises(TargetSafetyError, match="Alternate hostaddr override '10.0.0.1' is rejected"):
        validate_isolated_target(url, env={})


def test_reject_multiple_hostaddr():
    url = f"{VALID_DISPOSABLE_URL}?hostaddr=127.0.0.1,10.0.0.1"
    with pytest.raises(TargetSafetyError, match="Multiple hostaddr targets are rejected"):
        validate_isolated_target(url, env={})


# 5. Port validation
def test_reject_wrong_port():
    url = "postgresql://test_user:test_password@127.0.0.1:5432/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Port must be isolated port '5433'"):
        validate_isolated_target(url, env={})


def test_reject_missing_port():
    url = "postgresql://test_user:test_password@127.0.0.1/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Missing required port parameter"):
        validate_isolated_target(url, env={})


def test_reject_multiple_ports():
    url = "postgresql://test_user:test_password@127.0.0.1:5433,5432/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Multiple host or port targets are rejected"):
        validate_isolated_target(url, env={})


# 6. Database name validation
def test_reject_wrong_database():
    url = "postgresql://test_user:test_password@127.0.0.1:5433/test_triage_desk"
    with pytest.raises(TargetSafetyError, match="Database name must be 'triage_events_disposable'"):
        validate_isolated_target(url, env={})


def test_reject_missing_database():
    url = "postgresql://test_user:test_password@127.0.0.1:5433/"
    with pytest.raises(TargetSafetyError, match="Missing required database name parameter"):
        validate_isolated_target(url, env={})


# 7. User validation
def test_reject_wrong_user():
    url = "postgresql://postgres:test_password@127.0.0.1:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Database user must be 'test_user'"):
        validate_isolated_target(url, env={})


def test_reject_missing_user():
    url = "postgresql://127.0.0.1:5433/triage_events_disposable"
    with pytest.raises(TargetSafetyError, match="Missing required user parameter"):
        validate_isolated_target(url, env={})


# 8. Service and servicefile rejection in conninfo
def test_reject_service_parameter():
    url = f"{VALID_DISPOSABLE_URL}?service=primary"
    with pytest.raises(TargetSafetyError, match="Service-based connection configuration is rejected"):
        validate_isolated_target(url, env={})


def test_reject_servicefile_parameter():
    url = f"{VALID_DISPOSABLE_URL}?servicefile=/tmp/pg_service.conf"
    with pytest.raises(TargetSafetyError, match="Servicefile connection configuration is rejected"):
        validate_isolated_target(url, env={})


# 9. Ambient libpq environment variable inspection
@pytest.mark.parametrize(
    "env_key,env_val,match_msg",
    [
        ("PGSERVICE", "myservice", "Ambient environment variable PGSERVICE is set"),
        ("PGSERVICEFILE", "/tmp/service.conf", "Ambient environment variable PGSERVICEFILE is set"),
        ("PGSYSCONFDIR", "/etc/postgresql", "Ambient environment variable PGSYSCONFDIR is set"),
        ("PGHOST", "remote.host.com", "Ambient environment variable PGHOST conflicts with required host '127.0.0.1'"),
        ("PGHOSTADDR", "10.0.0.5", "Ambient environment variable PGHOSTADDR conflicts with required host '127.0.0.1'"),
        ("PGPORT", "5432", "Ambient environment variable PGPORT conflicts with required port '5433'"),
        ("PGDATABASE", "production", "Ambient environment variable PGDATABASE conflicts with required database 'triage_events_disposable'"),
        ("PGUSER", "postgres", "Ambient environment variable PGUSER conflicts with required user 'test_user'"),
    ],
)
def test_reject_conflicting_ambient_env(env_key, env_val, match_msg):
    with pytest.raises(TargetSafetyError, match=match_msg):
        validate_isolated_target(VALID_DISPOSABLE_URL, env={env_key: env_val})


def test_accept_matching_ambient_env():
    matching_env = {
        "PGHOST": "127.0.0.1",
        "PGHOSTADDR": "127.0.0.1",
        "PGPORT": "5433",
        "PGDATABASE": "triage_events_disposable",
        "PGUSER": "test_user",
    }
    params = validate_isolated_target(VALID_DISPOSABLE_URL, env=matching_env)
    assert params["host"] == "127.0.0.1"


# 10. Unexpected connection parameters
def test_reject_unexpected_connection_parameter():
    url = f"{VALID_DISPOSABLE_URL}?target_session_attrs=primary"
    with pytest.raises(TargetSafetyError, match="Unexpected connection parameter 'target_session_attrs' is rejected"):
        validate_isolated_target(url, env={})


# 11. Cloud infrastructure keyword rejection
@pytest.mark.parametrize(
    "cloud_url",
    [
        "postgresql://test_user:pw@aws-0-ap-southeast-1.pooler.supabase.com:5433/triage_events_disposable",
        "postgresql://test_user:pw@127.0.0.1:5433/triage_events_disposable?application_name=neon_app",
        "postgresql://test_user:pw@127.0.0.1:5433/triage_events_disposable?application_name=rds_monitoring",
    ],
)
def test_reject_cloud_infrastructure_keywords(cloud_url):
    with pytest.raises(TargetSafetyError, match="references non-disposable infrastructure|Host must be exactly"):
        validate_isolated_target(cloud_url, env={})


# 12. Password and secret confidentiality: passwords must NEVER appear in error messages or exceptions
@pytest.mark.parametrize(
    "invalid_url_with_secret",
    [
        "postgresql://test_user:super_secret_high_entropy_secret_12345@192.168.1.55:5433/triage_events_disposable",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@127.0.0.1:5432/triage_events_disposable",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@127.0.0.1:5433/wrong_db",
        "postgresql://wrong_user:super_secret_high_entropy_secret_12345@127.0.0.1:5433/triage_events_disposable",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@127.0.0.1:5433/triage_events_disposable?service=foo",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@127.0.0.1:5433/triage_events_disposable?hostaddr=9.9.9.9",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@127.0.0.1:5433/triage_events_disposable?target_session_attrs=read-write",
        "postgresql://test_user:super_secret_high_entropy_secret_12345@aws-0-db.pooler.supabase.com:5433/triage_events_disposable",
        ":::malformed:::super_secret_high_entropy_secret_12345:::",
    ],
)
def test_password_never_leaked_in_exception(invalid_url_with_secret):
    secret = "super_secret_high_entropy_secret_12345"
    with pytest.raises(TargetSafetyError) as excinfo:
        validate_isolated_target(invalid_url_with_secret, env={})
    error_message = str(excinfo.value)
    assert secret not in error_message, f"Secret leaked in error message: {error_message}"
    assert excinfo.value.__cause__ is None, "Chained exception cause must be suppressed to prevent secret leakage"
