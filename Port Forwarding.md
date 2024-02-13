## Steps:


1. open two terminals (A, B)
2. Terminal A: 

```
# connect to server
ssh username@hostname

# go to home directory repos
# example:
cd /bioinf/home/mschecht/post_msc

# Now on remote server via ssh
jupyter lab --no-browser --port=8888
```
5. Terminal B:

```
# On local computer:
ssh -Y -N -L localhost:888X:localhost:8888 username@hostname
```
6. Copy link in stdout of step 4 and paste into web browser
7. Now on remote server via ssh
```
jupyter lab --no-browser --port=8888
```
