import tensorflow_datasets as tfds

beam = tfds.core.lazy_imports.apache_beam

dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions()
)
wp = tfds.load("wikipedia")
wp.download_and_prepare(download_config=dl_config)