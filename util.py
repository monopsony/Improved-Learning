import subprocess,pickle,os,sys,getopt,para,shutil
from types import ModuleType 



path_separator=((os.name=='nt') and "\\") or  "/" 
dir_path=os.path.dirname(os.path.realpath(__file__))+path_separator
custom_files_dir=dir_path+'custom_files'+path_separator
sys.path.append(custom_files_dir)


class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

try:
  basestring
except NameError:
  basestring = str

output_file_name="out.txt"
output_file=open(output_file_name,"w+")

def print_warning(s):
	print(bcolors.WARNING + s + bcolors.ENDC,flush=True)
	output_file.write(s)

def print_error(s):
	print(bcolors.FAIL + s + bcolors.ENDC,flush=True)
	output_file.write(s)
	sys.exit(2)

def print_blue(s):
	print(bcolors.OKBLUE + s + bcolors.ENDC,flush=True)
	output_file.write(s)

def print_debug(s):
	if not isinstance(s,basestring):
		s=str(s)
	print(bcolors.OKGREEN + s + bcolors.ENDC,flush=True)
	output_file.write(s)

print_debug("IN UTIL")

def print_help():
	print("THIS IS THE HELP FUNCTION")

def find_function(name,*args):
	if name is None:
		print_error('Error in find_function: name is None. Aborted.')
	if not isinstance(name,basestring):
		print_error('Error in find_function: {} not a string. Aborted.'.format(name))

	if len(args)<2:
		print_error('find_function called with only {} arguments (at least 2 needed). Aborted.'.format(len(args)+1))

	for a in args:
		if isinstance(a,dict):
			f=((name in a) and a[name]) or False
			if callable(f):
				return f

		elif isinstance(a,ModuleType):
			f=getattr(a,name,False)
			if callable(f):
				return f

		else:
			print_warning('Argument {} passed to find_function: not a valid type (dict or module)'.format(a))

	print_error('No callable function called {} found in find_function. Aborted.'.format(name))

def generate_custom_args(db,args):
	
	if not (isinstance(args,list) or isinstance(args,tuple)):
		print_error('Invalid variable type for args in generate_custom_args. Must be tuple or list. Aborted')

	a=[]
	for i in args:
		if isinstance(i,basestring):
			n=i.find('func:')
			if n!=-1:
				f_name=i[n+5:]
				f=find_function(f_name,para,globals())
				a.append(f(db))
			else:
				a.append(i)

		else:
			a.append(i)

	return a

def run_custom_file(db,file,args):

	file_name,ext=os.path.splitext(file)
	path=custom_files_dir+file

	args=generate_custom_args(db,args)

	if ext=='.py':
		file=__import__(file_name)
		file.run(*args)
	elif ext=='.sh':
		call=[path]+[str(x) for x in args] #bash script needs it all to be strings ofc
		subprocess.call(call)
	else:
		print_error("File type {} not recognised in run_custom_file. Aborted.".format(ext))









