from distutils.core import setup

setup(
  name = 'HR_Neural_Networks',         # How you named your package folder (MyLib)
  packages = ['HR_Neural_Networks'],   # Chose the same as "name"
  version = '1.0.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Holistic Robust Neural Networks',   # Give a short description about your library
  author = 'Amine Bennouna, Bart Van Parys, Ryan Lucas',                   # Type in your name
  author_email = 'ryanlu@mit.edu',      # Type in your E-Mail
  url = 'https://github.com/RyanLucas3/HR_Neural_Networks',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/RyanLucas3/HR_Neural_Networks/archive/refs/tags/1.0.3.tar.gz',
  keywords = ['Neural Networks', 'Robustness', 'Machine Learning', "Data Science", "Adversarial Attacks"],   # Keywords that define your package best
  install_requires=[ # I get to this in a second
          'mosek',
          'torchattacks',
          'numpy',
          'cvxpy',
          'torch'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)