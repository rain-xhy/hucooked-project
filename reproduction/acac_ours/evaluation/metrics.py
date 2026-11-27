from statistics import mean


def summarize(final_infos):
    episodes = [info["metrics"] for info in final_infos]
    if not episodes:
        raise ValueError("At least one episode is required")

    attempts = sum(episode["joint_attempt_steps"] for episode in episodes)
    successes = sum(episode["joint_successes"] for episode in episodes)
    return {
        "episodes": len(episodes),
        "mean_meals": mean(episode["meals_completed"] for episode in episodes),
        "mean_team_score": mean(episode["team_score"] for episode in episodes),
        "sync_success_percent": 100 * successes / attempts if attempts else 0.0,
        "mean_wait_steps": mean(episode["joint_wait_steps"] for episode in episodes),
    }
