The files which contain the code written by me are football.gfootball.examples.qmix for the QMIX algorithm, and football.gfootball.examples.a2c for the A2C one.
In the Qmix one, the GroupAgentsWrapperFootball class was initially a class part of the rllib library which had to be modified (observation_space_sample and action_space_sample)
had to be overwritten. The rest is done by me. In the a2c file, the environment wrapper was given in the example, and I didn't have to change it.
