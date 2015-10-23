set message = $argv[1]
git add -A
git commit -m "$message"
git push -u DSproject master # push the master barnch to DSproject repo.
