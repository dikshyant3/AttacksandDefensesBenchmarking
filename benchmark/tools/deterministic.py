class DeterministicTools:
    """Every tool call with the same input MUST return the same output, always.

    No real network calls, no real deployments — these are simulations.
    """

    def run_tests(self, target: str) -> dict:
        return {"status": "passed", "target": target, "failures": 0}

    def git_push(self, branch: str, commit_msg: str) -> dict:
        return {"status": "success", "branch": branch, "commit_hash": "deterministic_abc123"}

    def deploy(self, environment: str) -> dict:
        return {"status": "deployed", "environment": environment, "version": "v1.0.0-test"}

    def rollback(self, environment: str) -> dict:
        return {"status": "rolled_back", "environment": environment}
