from setuptools import setup, find_packages

# Read README content
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

config = {
    'description': 'An open-source Python toolkit offering a collection of efficient, easy-to-use functions for seismic data analysis.',
    'author': 'Gabriele Paoletti',
    'url': 'https://github.com/gabrielepaoletti/seismoviz',
    'download_url': 'https://github.com/gabrielepaoletti/seismoviz',
    'author_email': 'gabriele.paoletti@uniroma1.it',
    'version': '0.0.1',
    'python_requires': '>=3.11',
    'install_requires': ['cartopy', 'holoview', 'matplotlib', 'numpy', 'pandas', 'panel', 'pyproj', 'scipy', 'srtm'],
    'packages': find_packages(),
    'name': 'seismoviz',
    'license': 'MIT',
    'keywords': 'seismology earthquake geophysics data-analysis',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
}

setup(**config)