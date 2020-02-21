import tensorflow as tf

log_dir = 'tensorboard_v2'
summary_writer = tf.summary.create_file_writer(log_dir) 
tf.summary.trace_on(graph=True, profiler=True) 

x = tf.Variable(3.)
@tf.function
def grad():
    with tf.GradientTape() as tape:
        y = tf.square(x)
    y_grad = tape.gradient(y, x)
    return y_grad

y_grad = grad()
print(y_grad)
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)

