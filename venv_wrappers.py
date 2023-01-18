from abc import ABCMeta, abstractmethod
from types import FunctionType
import envpool

import flax
import jax.numpy as jnp
import jax

from RunningMeanStd import RunningMeanStd

class VectorEnvWrapper:
    def __init__(self,envs,wrappers):
        self.envs=envs
        self.wrappers=wrappers
        self._handle, self._recv, self._send, self._step=self.envs.xla()

    def reset(self):
        result=self.envs.reset()
        handles=[self._handle]
        for wrapper in self.wrappers:
            handle,result=wrapper.reset(result)
            handles+=[handle]
        return handles,result

    def xla(self):
        def _apply_handle(ret,x):
            f,handle=x
            newhandle,ret=f(handle,ret)
            return ret,newhandle

        @jax.jit
        def recv(handles: jnp.ndarray):
            _handle,ret = self._recv(handles[0])
            new_handles=[]
            #reversed
            for handle in reversed(handles[1:]):
                handle,ret=handle.recv(ret)
                new_handles+=[handle]
            return [_handle]+list(reversed(new_handles)), ret

        @jax.jit
        def send(handle: jnp.ndarray,action,env_id= None):
            for wrapper in self.wrappers:
                action=wrapper.send(action)
            return [self._send(handle[0],action,env_id)]+handle[1:]

        def step(handle,action,env_id=None) :
            return recv(send(handle, action, env_id))

        return self._handle,recv, send, step

@flax.struct.dataclass
class EnvWrapper:
    def recv(self,ret) :
        return self, ret

    def reset(self,ret):
        return self, ret
    
    def send(self,action):
        return action

@flax.struct.dataclass
class VectorEnvNormObs(EnvWrapper):
    obs_rms:RunningMeanStd=RunningMeanStd()
    def recv(self,ret) :
        next_obs, reward, next_done,next_truncated, info= ret
        obs_rms=self.obs_rms.update(next_obs)
        return self.replace(obs_rms=obs_rms), (obs_rms.norm(next_obs), reward, next_done,next_truncated, info)

    def reset(self,ret):
        obs,info = ret
        obs_rms=self.obs_rms.update(obs)
        obs=obs_rms.norm(obs).astype(jnp.float32)
        return self.replace(obs_rms=obs_rms), (obs,info)




@flax.struct.dataclass
class VectorEnvClipAct(EnvWrapper):
    action_low:jnp.array
    action_high:jnp.array
    def send(self,action):
        action_remap=jnp.clip(action, -1.0, 1.0)
        action_remap=(self.action_low+(action_remap+1.0)*(self.action_high-self.action_low)/2.0).astype(jnp.float64)
        return action

if __name__ == "__main__":
    envs = envpool.make(
        "HalfCheetah-v3",
        env_type="gym",
        num_envs=2,
        seed=0,
    )
    wrappers=[VectorEnvNormObs(),VectorEnvClipAct(envs.action_space.low,envs.action_space.high)]
    a=VectorEnvWrapper(envs,wrappers)
    handle, recv, send, step_env = a.xla()
    handle,s=a.reset()
    send(handle,jnp.array([[0.]*6]*2))
    # print(jax.make_jaxpr(send)(handle,jnp.array([[0.]*6]*2)))
    recv(handle)
    print(step_env(handle,jnp.array([[0.]*6]*2)))