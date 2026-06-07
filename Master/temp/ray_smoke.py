import ray

ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)


@ray.remote
def f(x):
    return x * x


out = ray.get([f.remote(i) for i in range(4)])
print("ray remote tasks OK:", out)
ray.shutdown()
print("ray.init + shutdown OK")
