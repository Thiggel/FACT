import time, datetime

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')


# NUM_PAR_SAMPLER = args_global.num_cpu_core
NUM_PAR_SAMPLER = 20
SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER) # round up division

EVAL_VAL_EVERY_EP = 1       # get accuracy on the validation set every this # epochs


f_mean = lambda l: sum(l)/len(l)

DTYPE = "float32"  # if args_global.dtype=='s' else "float64"      # NOTE: currently not supporting float64 yet
