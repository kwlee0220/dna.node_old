from setuptools import setup, find_packages
import dna

setup( 
    name = 'dna.node',
    version = dna.__version__,
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
			'dna_node_processor = scripts.dna_node_processor:main',
			'dna_node_processor_client = scripts.dna_node_processor_client:main',
			'dna_publish_events = scripts.dna_publish_events:main',
			'dna_publish_event_server = scripts.dna_publish_event_server:main',
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
        'tqdm>=4.41.0',
        'Shapely',
        'easydict',
        'pyyaml',
        'gdown',

				# geodesic transformation
				'pyproj',

        # rabbitmq
        'pika',

        # yolov5
        'ipython',
        'psutil',

        # siammot
        'gluoncv',
        'mxnet',
        'imgaug',
    ],
    packages = find_packages(),
    # package_data = {'conf': ['etri_051.yaml']},
    python_requires = '>=3.8',
    zip_safe = False
)
