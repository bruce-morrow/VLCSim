from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="VLCSim",
    version="1.0",
    description="A Visible Light Communications (VLC) simulator",
    author="Bruce Morrow",
    author_email="bruce.morrow@hotmail.com",
    url="https://github.com/bruce-morrow/VLCSim.git",
    license="gpl-3.0",
    install_requires=required,
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Communications :: Email',
    ]
)
