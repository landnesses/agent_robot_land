from distutils.core import setup
# from platform import version
# from catkin_pkg.python_setup import generate_distutils_setup
# from torch._C import ScriptModuleSerializer
# setup_args = generate_distutils_setup(
#          version='0.0.0',
#          scripts=['scripts/v5detecttalker.py','scripts/v5detectlisener.py'],
#          packages=['darknet_ros'],
#         #  package_dir={' ':'src'},
#          package_dir={'':'scripts'}
#          )
# setup(**setup_args)

setup(
         version='0.0.0',            #根据xml文件里对应来修改的：  <version>0.0.0</version>
         scripts=['scripts/v5detecttalker.py','scripts/v5detectlisener.py','scripts/talker.py'],
         packages=['darknet_ros'],
         package_dir={'':'scripts'}
)