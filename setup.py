from setuptools import setup, find_packages

setup( 
    name = 'dna.node',
    version = '0.0.3',
    description = 'DNA framework',
    author = 'Kang-Woo Lee',
    author_email = 'kwlee@etri.re.kr',
    url = 'https://github.com/kwlee0220/dna.node',
	entry_points={
		'console_scripts': [
			'dna_node_show = scripts.dna_node_show:main',
			'dna_node_detect = scripts.dna_node_detect:main',
			'dna_node_track = scripts.dna_node_track:main',
			'dna_node = scripts.dna_node:main',
			'dna_publish_events = scripts.dna_publish_events:main',
		],
	},
    install_requires = [
        'numpy>=1.18.5',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',

        'opencv-python>=4.1.2',
        'kafka-python',

        'omegaconf>=2.1.2',
        'tqdm',
        'Shapely',
        'easydict',
        'pyyaml',
        'gdown',
    ],
    packages = find_packages(),
    package_data = {'conf': ['etri_051.yaml']},
    python_requires = '>=3.8',
    zip_safe = False
)
