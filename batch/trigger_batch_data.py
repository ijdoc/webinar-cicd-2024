import requests
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Trigger GitHub Actions workflow with arguments."
    )
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
    parser.add_argument(
        "--type",
        type=str,
        default="production",
        choices=["training", "production"],
        help="Type of data",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=200,
        help="Total length of the history batch in days",
    )
    parser.add_argument(
        "--stride-days",
        type=int,
        default=1,
        help="Number of days to stride between iterations",
    )
    parser.add_argument("--token", type=str, help="GitHub token with repo permissions")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository in the form owner/repo",
    )
    args = parser.parse_args()

    # Get the GitHub token from argument or environment variable
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GitHub token must be provided via --token argument or GITHUB_TOKEN environment variable"
        )

    # Split the repository into owner and repo
    try:
        owner, repo = args.repo.split("/")
    except ValueError:
        raise ValueError("Repository must be in the form owner/repo")

    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"

    headers = {
        "Accept": "application/vnd.github.everest-preview+json",
        "Authorization": f"token {token}",
    }

    data = {
        "event_type": "prod_data_batch",  # Must match the type defined in the workflow
        "client_payload": {
            "iteration": args.iteration,
            "type": args.type,
            "history_days": args.history_days,
            "stride_days": args.stride_days,
        },
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 204:
        print("Workflow triggered successfully.")
        print(response.text)
    else:
        print(f"Failed to trigger workflow: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
