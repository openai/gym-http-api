# retro_contest_agent

My submission for OpenAI Retro Contest, based of OpenAI PPO2.

https://contest.openai.com/details

# Description

The approach I followed is based on OpenAI PPO2 baseline implementation and is named PPO2ttifrutti after it. The meta-learning phase consists in a joint training as described in the technical report with hyperparameters tuned for efficient use of my hardware and best accuracy. This setup was chosen to be able to rapidly prototype different variants of PPO for Sonic. I tried many variations inspired from papers, e.g. using experience replay, or from usual approaches in supervised learning, such as data augmentation. In the final version of the meta-learning phase the inputs were modified uniformly within a rollout by adding a constant value picked at random to every pixels. Regarding the learning phase, I tuned the hyperparameters to fine-tune a fast as possible the model trained during meta-learning. I noticed that the first 150K time steps were subject to high variations in terms of score: initial performance decreased significantly before coming back to its original level, so I tuned the learning rate to prevent this phenomenon and avoid wasting the first 15% of every run. The final model is trained on all the Sonic save states provided for the contest, as well as levels from Sonic 1 & 2 on Master System and Sonic Advance on Gameboy Advance, which unfortunately didn't seem to yield significant improvement compared to training on the original training set.

I focused on the simplest ideas I could think of and because experimenting was costly in time I used only 3 runs to obtain a performance estimate for the learning phase, and sometimes only one run for the most expensive meta-learning phases. The general idea was: if it doesn't decrease performance, keep it in the hope it would improve test time performance. I departed from this rule a few times when significantly more experimentation was required to achieve a good level of confidence that it wasn't degrading performance in some cases.

The final network is based on the default CNN provided along with the PPO2 baseline but uses depthwise separable convolutions and more inputs for the fully connected layer. The former change was originally introduced to decrease the number of parameters and train faster, but in the final network the number of parameters is higher. The network uses a stack of 4 96x96 grayscale images. Many tries using color, a different stack size and different number of layers resulted in mixed results. I tried using more complex ideas, e.g. inverted residuals (MobileNetV2) and different optimizers (either available in TensorFlow or my own implementation) but these ideas didn't give good results in the time-frame I allowed myself. Another example of something that I had to drop quickly was a model better generalizing to levels it has never seen before (early scores at test time were higher) but also a lot slower to learn in the long run and giving poor results after 1M timesteps (the number of timesteps used a test time).

Compared to the original PPO2 baseline, besides starting from a pretrained model instead of learning from scratch, at test time the agent uses 96x96 inputs, a different CNN, smaller batch size and learning rate, and use reward reshaping. The checkpoint used for the final submission was generated after 725 updates.

This work is based on code provided by OpenAI, which can be found here:

https://github.com/openai/retro

https://github.com/openai/baselines

https://github.com/openai/retro-baselines
