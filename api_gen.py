import namex

from keras_aug._src.version import __version__

namex.generate_api_files(package="keras_aug", code_directory="_src")

# Add version string

with open("keras_aug/__init__.py", "r") as f:
    contents = f.read()
with open("keras_aug/__init__.py", "w") as f:
    contents += f'__version__ = "{__version__}"\n'
    f.write(contents)
