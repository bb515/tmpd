"""Image restoration experiments."""
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["eval", "eval_from_file", "sample", "inpainting", "super_resolution", "deblur"],
                  "Running mode: sample, inpainting, super_resolution or deblur")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    if FLAGS.mode == "sample":
        run_lib.sample(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval":
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "eval_from_file":
        run_lib.evaluate_from_file(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "dps_hyperparameter_search":
        run_lib.dps_hyperparameter_search(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "inpainting":
        run_lib.inpainting(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "super_resolution":
        run_lib.super_resolution(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "deblur":
        run_lib.deblur(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)


if __name__ == "__main__":
    app.run(main)
