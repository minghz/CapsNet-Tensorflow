import tensorflow as tf

def conv_layer(input, in_channels, num_outputs,
               kernel_size, stride, padding, act=tf.nn.relu):
    W = tf.get_variable('W',
        initializer=tf.truncated_normal([kernel_size, kernel_size,
            in_channels, num_outputs], stddev=0.1))
    conv = act(tf.nn.conv2d(input, W,
        strides=[1, stride, stride, 1], padding=padding))

    tf.summary.histogram('W', W)
    tf.summary.histogram('conv', conv)

    return conv


def to_fixed_point(x, scope):
  """Helper method to convert tensors to fixed point accuracy

  Args:
    x: input tensor
    scope: variable scope that is being converted
  Returns:
    fixed point accuracy equivalent tensor
  """
  with tf.variable_scope(scope):
    fix_def = tf.get_variable('fix_def', initializer=[1, 1], dtype=tf.int32, trainable=False)
    acc = tf.get_variable('acc', initializer=[0., 0.], trainable=False)

  fixed_x = reshape_fix(x, fix_def, acc)
  fixed_point_conversion_summary(x, fixed_x, fix_def, acc)

  return fixed_x


def fixed_point_conversion_summary(x, fixed_x, fix_def, acc):
  """Helper to create summaries for fixed point conversion steps.

  Creates a summary that provies a histogram of resulting tensor
  Creates a summary that provides the percentage innacuracy of the conversion

  Args:
    x: original tensor
    fixed_x: Resulting tensor
    fix_def: fix point definition for this conversion
    acc: accuracy array
  Returns:
    nothing
  """
  with tf.variable_scope('fix_def'):
    tf.summary.scalar('digit bits', fix_def[0])
    tf.summary.scalar('fraction bits', fix_def[1])

  with tf.variable_scope('acc'):
    tf.summary.scalar('percentage clip', (acc[0]))
    tf.summary.scalar('percentage under tolerance', (acc[1]))

  tf.summary.histogram('original', x)
  tf.summary.histogram('fixed', fixed_x)

