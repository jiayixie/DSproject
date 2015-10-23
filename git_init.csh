# initiate a repo here, only use once when building new repo
set message = $argv[1]
git init
git config --global user.name "jiayixie"
git config --global user.email jiayi.seis@gmail.com
git add -A
git commit -m "$message" 
git remote add DSproject git@github.com:jiayixie/DSproject.git # use a name to represent the remote URL repo

