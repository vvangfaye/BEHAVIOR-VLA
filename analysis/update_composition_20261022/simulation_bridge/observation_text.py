"""Match the evaluation server's whole-task language conditioning."""


def resolve_prompt(observation, task_prompt):
    prompt = observation['prompt'] if 'prompt' in observation else task_prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('A nonempty task instruction is required')
    return prompt
