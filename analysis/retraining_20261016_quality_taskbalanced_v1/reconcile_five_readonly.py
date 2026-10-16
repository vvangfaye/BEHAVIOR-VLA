"""Read existing five-job reports and compare finished episodes; never submit or stop."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys

from observe_five_batches import CAMPAIGN, ROOT, controller, snapshot


def main():
    observation = snapshot()  # Verifies exact existing UIDs, ownership, images and resources.
    plan = controller.load(CAMPAIGN / 'plan.json')
    client = controller.aoss_client()
    ids = [f'candidate-{i:02d}' for i in range(20)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        reports = list(pool.map(lambda key: controller.get_report(client, plan, key), ids))
    result = dict(observed_at=observation['observed_at'], cloud_read_only=True,
                  reports_present=sum(r is not None for r in reports),
                  accepted_report_claims=sum(bool(r and r.get('state') == 'accepted') for r in reports),
                  all_jobs_terminal=all(r['native']['state'] in controller.TERMINAL
                                        for r in observation['rows']))
    # This validates protocol, bundle, checkpoint, episode identity and record fields.
    checked = subprocess.run([sys.executable, '-B', str(ROOT / 'compare_partial_fixed20.py')],
                             capture_output=True, text=True, check=True)
    result['comparison'] = json.loads(checked.stdout)
    controller.save(CAMPAIGN / 'readonly_reconciliation.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
