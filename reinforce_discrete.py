# -*- coding: UTF-8 -*-
"""
filename: REINFORCE_discrete.py
function: the REINFORCE algorithm for discrete variables
    date: 2017/8/7
  author:
________                            ____.__
\______ \   ____   ____   ____     |    |__| ____   ____ _____    ____
 |    |  \_/ __ \ /    \ / ___\    |    |  |/    \_/ ___\\__  \  /    \
 |    `   \  ___/|   |  / /_/  /\__|    |  |   |  \  \___ / __ \|   |  \
/_______  /\___  |___|  \___  /\________|__|___|  /\___  (____  |___|  /
        \/     \/     \/_____/                  \/     \/     \/     \/

    　　 へ　　　　　／|
　　/＼7　　　 ∠＿/
　 /　│　　 ／　／
　│　Z ＿,＜　／　　 /`ヽ
　│　　　　　ヽ　　 /　　〉
　 Y　　　　　`　 /　　/
　ｲ●　､　●　　⊂⊃〈　　/
　()　 へ　　　　|　＼〈
　　ｰ ､_　 ィ　 │ ／／
　 / へ　　 /　ﾉ＜| ＼＼
　 ヽ_ﾉ　　(_／　 │／／
　　7　　　　　　　|／
　　＞―r￣￣`ｰ―＿
"""
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

import math

# set ctx
data_ctx = mx.cpu()
model_ctx = mx.cpu()


class Policy(gluon.Block):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(hidden_size)
            self.dense1 = gluon.nn.Dense(num_outputs)

    def forward(self, inputs):
        x = inputs
        x = nd.relu(self.dense0(x))
        action_scores = self.dense1(x)

        return nd.softmax(action_scores)


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.model.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=model_ctx)
        self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': 0.001})

    def select_action(self, state):
        with autograd.record():
            probs = self.model(state.as_in_context(model_ctx))
            action = nd.random.multinomial(probs)
            prob = probs[:, action[0]].reshape((1, -1))
            log_prob = nd.log(prob)
            entropy = - (probs * nd.log(probs)).sum()

        return action[0], log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        with autograd.record():
            R = nd.zeros((1, 1))
            loss = 0
            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                loss = loss - (log_probs[i] * R).sum() - (0.0001 * entropies[i]).sum()
            # loss = loss / len(rewards)
        self.model.collect_params().zero_grad()
        loss.backward()
        grads = [i.grad(data_ctx) for i in self.model.collect_params().values()]
        # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
        # 因此我们将clipping_norm乘以num_steps和batch_size。
        gluon.utils.clip_global_norm(grads, 40)
        self.optimizer.step(batch_size=len(rewards))


"""
░░░░░░░░░▄░░░░░░░░░░░░░░▄░░░░
░░░░░░░░▌▒█░░░░░░░░░░░▄▀▒▌░░░
░░░░░░░░▌▒▒█░░░░░░░░▄▀▒▒▒▐░░░
░░░░░░░▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐░░░
░░░░░▄▄▀▒░▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐░░░
░░░▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌░░░ 
░░▐▒▒▒▄▄▒▒▒▒░░░▒▒▒▒▒▒▒▀▄▒▒▌░░
░░▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐░░
░▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌░
░▌░▒▄██▄▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌░
▀▒▀▐▄█▄█▌▄░▀▒▒░░░░░░░░░░▒▒▒▐░
▐▒▒▐▀▐▀▒░▄▄▒▄▒▒▒▒▒▒░▒░▒░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▐░
░▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒░▒░▒░▒░▒▒▒▌░
░▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▄▒▒▐░░
░░▀▄▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▄▒▒▒▒▌░░
░░░░▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀░░░
░░░░░░▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀░░░░░
░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▀▀░░░░░░░░
"""
