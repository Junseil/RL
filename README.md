# 강화학습 알고리즘들의 구현과 커스텀

23.02.28 시정 모든 라이브러리 최신 버전에서 작동 확인

## 0. DQN_papertest

논문에 활용했던 코드... 현재 Gym 에서의 함수가 일부 변경되어 수정 필요

## 1. RL_nonDL
	1. MDP를 통한 policy iteration, value iteration
	2. MDP가 주어지지 않은 상황에서의 On-policy 방법인 MC control, n-step SARSA
	3. MDP가 주어지지 않은 상황에서의 Off-policy 방법인 Q learning

Fastcampus의 Gridworld 환경 소스 코드를 FrozenLake에 맞게 __변형 및 커스텀__

## 2. DRL
	1. Multi armed bandit의 epsilon greedy, softmax, context 버전
	2. Replay memory, target Q-network가 반영이 되지않은 naive한 DQN 버전
	3. REINFORCE, A2C, DDPG, PPO 지속적으로 업데이트 예정

Deep Reinfrocement Learning in Action 교재를 통해 클론 코딩 학습 및 __일부 커스텀__

23.03.01 ~ DL 학습 ing...
