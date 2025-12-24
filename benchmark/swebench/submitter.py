import requests
from typing import List, Dict


class SWEBenchAPIClient:
    def __init__(self, endpoint: str, timeout: int = 30):
        self.endpoint = endpoint
        self.timeout = timeout

    def submit(self, payload: Dict):
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def submit_batch(self, submissions: List[Dict]):
        results = []
        for payload in submissions:
            results.append(self.submit(payload))
        return results
