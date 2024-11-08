from model.DQN_model import dqn_model, eval_callback

dqn_model.learn(
    total_timesteps = 4_000_000,
    progress_bar = True,
    callback = eval_callback
)
dqn_model.save("dqn_snake")