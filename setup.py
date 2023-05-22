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
			'dna_show = scripts.dna_show:main',
			'dna_detect = scripts.dna_detect:main',
			'dna_track = scripts.dna_track:main',
			'dna_node = scripts.dna_node:main',
			'dna_node_processor = scripts.dna_node_processor:main',
			'dna_node_processor_client = scripts.dna_node_processor_client:main',
			'dna_import_tracklets = scripts.dna_import_tracklets:main',
			'dna_draw_trajs = scripts.dna_draw_trajs:main',
			# 'dna_publish_events = scripts.dna_publish_events:main',
			# 'dna_publish_event_server = scripts.dna_publish_event_server:main',
			# 'dna_sync_videos = scripts.dna_sync_videos:main',
			# 'dna_show_multiple_videos = scripts.dna_show_multiple_videos:main',
   
			# 'dna_gen_trainset = scripts.dna_gen_trainset:main',
			# 'dna_gen_trainset2 = scripts.dna_gen_trainset2:main',
			# 'dna_reduce_trainset = scripts.dna_reduce_trainset:main',
			# 'dna_assoc_tracklets = scripts.dna_assoc_tracklets:main',
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

        # PostgreSQL
        'psycopg2',

        # rabbitmq
        'pika',

        # yolov5
        'ipython',
        'psutil',
        
        # protobuf
        'protobuf',
        
        # machine-learning
        'scikit-learn',
        'matplotlib',

        # siammot
        # 'gluoncv',
        # 'mxnet',
        # 'imgaug',
    ],
    packages = find_packages(),
    package_dir={'conf': 'conf'},
    package_data = {
        'conf': ['logger.yaml']
    },
    python_requires = '>=3.10',
    zip_safe = False
)
