import os
cwd = os.getcwd()[:-10]  # remove '/notebooks' (i.e. the last 10 characters)
tgt_dir = os.path.join(cwd, 'utils') + os.sep
ln_name = '.' + os.sep + 'utils'
os.system("ln -s \"{}\" {}".format(tgt_dir, ln_name))
print("Made symbolic link")
