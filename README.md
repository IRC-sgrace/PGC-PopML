# PGC-PopML
Machine Learning for Population Genetics: A Project with PGC IMBUE
I built a web app with a machine learning backend to deploy the population assignment machine we built using data collected from this research: https://www.sciencedirect.com/science/article/abs/pii/S0165783618302625


Here are the steps to run the code in localhost.

# Step 1: Activate Virtual Environment
I used a machine running on Windows 10 so all these instructions have been tested on a Windows machine as of writing. 

The steps described below are the same from the python documentation as noted here: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

On the command line (press the windows button then type CMD), run the following commands
>> python -m pip install --upgrade pip
>> python -m pip install --user virtualenv
>> python -m venv env
>> .\env\Scripts\activate

Once the environment is activated, you should see something like this

(env) C:\Users\User> 

... or whatever the name of your current path is

While the environment is activated and assuming all necessary files (app.py) are in the same path, we can now run streamlit using this command

(env) C:\Users\User>streamlit run app.py

Then a message similar to this should appear:

You can now view your Streamlit app in your browser.

... and the corresponding Local and Network URL will also be printed on the terminal



