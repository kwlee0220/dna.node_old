from setuptools import setup, find_packages
import dna

setup( 
    name = 'dna.tools',
    version = dna.__version__,
    description = 'Tools for DNA Node',
    author = 'Kang-Woo Lee',
    author_email = 'kwlee@etri.re.kr',
    url = 'https://github.com/kwlee0220/dna.node',
	entry_points={
		'console_scripts': [
			'dna_draw_trajs = scripts.dna_draw_trajs:main',
			'dna_replay_node_events = scripts.dna_replay_node_events:main',
   
			# 'dna_download_node_events = scripts.dna_download_node_events:main',
			# 'dna_show_multiple_videos = scripts.dna_show_multiple_videos:main',
			# 'dna_show_mc_locations = scripts.dna_show_mc_locations:main',
   
		],
	},
    install_requires = [
        'numpy>=1.18.5',
        'scipy',
        'omegaconf>=2.1.2',
		'tqdm>=4.41.0',
        
        # protobuf
        'protobuf',

        # geodesic transformation
        'pyproj',
        
        # Kafka
        'kafka-python',
        
        # OpenCv
        'opencv-python>=4.1.2',
    ],
    packages = find_packages(),
    package_dir={'conf': 'conf'},
    package_data = {
        'conf': ['logger.yaml']
    },
    python_requires = '>=3.10',
    zip_safe = False
)
