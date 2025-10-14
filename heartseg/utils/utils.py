from importlib.metadata import metadata

# Weights zip urls
# ZIP_URLS = dict([meta.split(', ') for meta in metadata('heartseg').get_all('Project-URL') if meta.startswith('Dataset')])
ZIP_URLS = dict(['https://github.com/pycadd/pycad-model-zoo/releases/download/v0.0.3/heart.zip'])

# Version
VERSION = metadata('heartseg').get('version')
