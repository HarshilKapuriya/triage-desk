"""End-to-End Demonstration of Phase 3 Financial Complaint Classification.

Demonstrates:
1. Ingest a real-world financial complaint via API key.
2. Observe ticket created with status 'new' and zero predictions.
3. Background worker claims job, executes cached LinearSVC inference on ticket body.
4. Worker atomically registers tenant model version, stores prediction, and transitions ticket.
5. Query ticket details via GET /v1/tickets/{id} to inspect predictions and margin scores.
"""

import json
import uuid
from fastapi.testclient import TestClient
import psycopg
from psycopg_pool import ConnectionPool

from app.main import app
from app.config import get_settings
import app.db as app_db
from app.worker.main import WorkerService
from tests.test_tickets_db import _create_tenant_and_api_key
from tests.test_auth_db import TEST_DB_URL, SAFETY_STATUS
from tests.test_ml_integration_db import _ml_test_db_lifespan


def run_demo() -> None:
    print("=" * 70)
    print("PHASE 3 FINANCIAL COMPLAINT CLASSIFICATION DEMONSTRATION")
    print("=" * 70)

    # Strictly require TEST_DATABASE_URL to protect development database from test fixture pollution
    if TEST_DB_URL is None:
        raise RuntimeError(
            f"Demo aborted: TEST_DATABASE_URL is not configured or failed safety checks ({SAFETY_STATUS}). "
            f"The demo strictly targets the isolated test database and will not target the development database."
        )

    db_url = TEST_DB_URL

    # Initialize connection pool for demo client
    pool = ConnectionPool(
        conninfo=db_url,
        min_size=1,
        max_size=5,
        open=True,
    )
    app_db.set_pool(pool)

    orig_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _ml_test_db_lifespan
    try:
        # 1. Tenant & API Key Provisioning (credentials held in memory only, never logged or printed)
        company_name = f"Demo Bank Client {uuid.uuid4().hex[:4]}"
        print(f"\n1. Provisioning tenant: '{company_name}'...")
        tenant_id, user_id, api_key, _ = _create_tenant_and_api_key(company_name)
        print(f"   - Tenant ID: {tenant_id}")
        print("   - API Key:   [Provisioned in memory, hidden from logs]")

        with TestClient(app) as client:
            # 2. Ingest Financial Complaint Ticket
            complaint_subject = "Dispute over uncredited checking account deposit and overdraft fees"
            complaint_narrative = (
                "I deposited a check for $1,200 into my personal checking account at the branch ATM on Monday. "
                "The teller stated funds would be available immediately, but the bank placed an extended hold "
                "without providing an exception hold notice. As a result, three routine household utility bills "
                "bounced, and the bank assessed $105 in non-sufficient funds (NSF) overdraft fees against my account."
            )

            print(f"\n2. Submitting financial complaint via POST /v1/tickets...")
            print(f"   - Subject: {complaint_subject}")
            print(f"   - Narrative snippet: {complaint_narrative[:110]}...")

            ingest_resp = client.post(
                "/v1/tickets",
                headers={"X-API-Key": api_key},
                json={
                    "subject": complaint_subject,
                    "body": complaint_narrative,
                    "external_id": f"complaint-ref-{uuid.uuid4().hex[:6]}",
                },
            )
            assert ingest_resp.status_code == 202, f"Expected 202, got {ingest_resp.status_code}: {ingest_resp.text}"
            ticket_data = ingest_resp.json()
            ticket_id = ticket_data["id"]

            print(f"   - Ingestion status: {ingest_resp.status_code} Accepted")
            print(f"   - Created Ticket ID: {ticket_id}")
            print(f"   - Initial Status:    {ticket_data['status']}")
            print(f"   - Human Category:    {ticket_data['category']}")
            print(f"   - Predictions count: {len(ticket_data.get('predictions', []))}")

            # 3. Worker Processing Cycle
            print(f"\n3. Running WorkerService cycle on background queue...")
            worker = WorkerService(db_url=db_url, worker_id="demo_worker_01")

            with psycopg.connect(db_url, autocommit=True) as conn:
                result = worker.run_one_cycle(conn, tenant_id=tenant_id)
            print(f"   - Worker execution result: '{result}'")

            # 4. Fetch Ticket Detail with Model Predictions
            print(f"\n4. Fetching updated ticket details via GET /v1/tickets/{ticket_id}...")
            detail_resp = client.get(
                f"/v1/tickets/{ticket_id}",
                headers={"X-API-Key": api_key},
            )
            assert detail_resp.status_code == 200, f"Expected 200, got {detail_resp.status_code}: {detail_resp.text}"
            updated_data = detail_resp.json()

            print(f"   - HTTP Status:        {detail_resp.status_code} OK")
            print(f"   - Current Status:     {updated_data['status']} (transitioned new -> open)")
            print(f"   - Human Category:     {updated_data['category']} (distinguished from prediction)")
            
            predictions = updated_data.get("predictions", [])
            print(f"   - Stored Predictions: {len(predictions)}")
            assert len(predictions) > 0, "No predictions returned on ticket detail!"

            pred = predictions[0]
            print("\n" + "-" * 50)
            print("PREDICTION METADATA & INFERENCE OUTPUT:")
            print("-" * 50)
            print(f"   Task:               {pred['task']}")
            print(f"   Predicted Category: {pred['predicted_label']}")
            print(f"   Confidence:         {pred['confidence']} (strictly NULL: LinearSVC margin distances are not probabilities)")
            print(f"   Model Version ID:   {pred['model_version_id']}")
            print(f"   Created At:         {pred['created_at']}")
            
            print("\n   Top Class Decision Margin Scores (Hyperplane Distances):")
            scores = pred.get("decision_scores") or {}
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (cls_name, margin) in enumerate(sorted_scores[:5], start=1):
                indicator = " <-- SELECTED" if cls_name == pred['predicted_label'] else ""
                print(f"     [{rank}] {cls_name:<52} : {margin:+7.4f}{indicator}")

            print("\n" + "=" * 70)
            print("DEMO COMPLETE: Financial complaint classified successfully.")
            print("=" * 70)
    finally:
        app.router.lifespan_context = orig_lifespan
        pool.close()
        app_db.set_pool(None)


if __name__ == "__main__":
    run_demo()
