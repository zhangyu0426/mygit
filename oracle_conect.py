# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:43:03 2016
连接数据
@author: shice
"""
#from tools import tools


class Oracle(object):
    
    def __init__(self):
        import cx_Oracle
        keys = ['user','pwd','ip','port','database']
        sql_dict = tools.read_config(path = '.tuixiang.conf',section = 'Oracle',keys = keys)
    
        db_list = []
        for key in keys:
            db_list.append(sql_dict.get(key))
            
        user, pwd, ip, port, database = db_list    
        connstr = '{}/{}@{}:{}/{}'.format(user, pwd, ip, port, database)
        self._conn = cx_Oracle.connect(connstr)
        self._cursor = self._conn.cursor()
        
    def select(self,sql):
        self._cursor.execute(sql)
        return self._cursor.fetchall()  
        
    def insert(self,sql):
        self._cursor.execute(sql)
        return self._cursor.fetchall()  
        
    def db_close(self):
        self._cursor.close
        self._conn.close()  
        
class SqlServel(object):
    
    def __init__(self):
        import pyodbc 
        keys = ['ip','port','database','user','pwd']
        sql_dict = tools.read_config(path = '.tuixiang.conf' ,section = 'SqlServer', keys = keys)
        
        db_list = []
        for key in keys:
            db_list.append(sql_dict.get(key))
        
        ip, port, database, user, pwd = db_list    
        connstr = 'DRIVER=FreeTDS;SERVER={};port={};DATABASE={};UID={};PWD={};'.format(ip, port, database, user, pwd)    
        self._conn = pyodbc.connect(connstr)
        self._cursor = self._conn.cursor()
            
    def select(self,sql):
        self._cursor.execute(sql)
        return self._cursor.fetchall()  
        
    def insert(self,sql):
        self._cursor.execute(sql)
        self._conn.commit()
        return self._cursor.rowcount
        
    def db_close(self):
        self._cursor.close
        self._conn.close()     

'''
https://gist.github.com/kimus/10012910
First of all, it just seems like doing anything with Oracle is obnoxiously painful for no good reason. It's the nature of the beast I suppose. cx_oracle is a python module that allows you to connect to an Oracle Database and issue queries, inserts, updates..usual jazz.

Linux

Step 1:

sudo apt-get install build-essential unzip python-dev libaio-dev
Step 2. Click here to download the appropriate zip files required for this. You'll need:

instantclient-basic-linux
instantclient-sdk-linux
Get the appropriate version for your system.. x86 vs 64 etc. Make sure you don't get version 12, since it's not supported by the cx_Oracle moduel yet.

Unzip the content in the same location, so you'll end up with a folder named: instantclient_11_2 (in my case) which will contain a bunch of .so and jar files.

just for ease of use I'll use $ORACLE_HOME which will basically point to the location where you unzipped your installclient folders.

export ORACLE_HOME=$(pwd)/instantclient_11_2
Step 3. create a symlink to your SO file.

cd $ORACLE_HOME
ln -s libclntsh.so.11.1   libclntsh.so  #the version number on your .so file might be different
Step 4. Update your /etc/profile or your ~/.bashrc

export ORACLE_HOME=/location/of/your/files/instantclient_11_2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ORACLE_HOME
Step 5: Edit /etc/ld.so.conf.d/oracle.conf

This is a new file, simple add the location of your .so files here, then update the ldpath using

sudo ldconfig
Step 6. Finaly just install cx_oracle module:

pip install cx_oracle
--------------------------------
 1. 下载instantclient basic和instantclient sdk，下载地址如下：
        instantclient basic 32位  64位
        instantclient sdk 32位  64位
    2. 解压下载的文件到/opt/oracle目录下，并创建软链接，命令如下：
1
2
3
4
5
6
7
# 解压文件
sudo mkdir -p /opt/oracle/
sudo unzip instantclient-sdk-linux32-10.2.0.3-20061115.zip
sudo unzip instantclient-sdk-linux32-10.2.0.3-20061115.zip
# 创建软连接
sudo ln -sf libclntsh.so.10.1 libclntsh.so
sudo ln -sf libocci.so.10.1 libocci.so
    3. 设置ORACLE_HOME等变量
    在.bashrc文件最后加入一下几行内容：
1
2
3
export ORACLE_HOME=/opt/opracle/instantclient_10_2
export LD_LIBRARY_PATH=$ORACLE_HOME
export TNS_ADMIN=$ORACLE_HOME/network/admin
    然后执行source ~/.bashrc
    4. 使用pip安装cx_Oracle
1
sudo pip install cx_Oracle
    问题1. DistutilsSetupError, cannot locate an Oracle software installation
    这个错误是在用pip安装cx_Oracle的时候出现的，大致的意思是说setup.py执行的时候找不到$ORACLE_HOME这个环境变量。但是在执行pip之前已经在bashrc中加入了环境变量，而且执行 env | grep ORACLE_HOME 也能找到相应的环境配置。
    纠结了半天之后，感觉可能是因为用户的问题，添加环境变量的时候是当前用户，而执行pip的时候是root用户。于是，执行 sudo env | grep ORACLE_HOME 命令，发现果然没有环境变量。找到原因之后就比较好办了：
1
2
3
4
5
sudo visudo
# 在 default specification 中加入一下两行
Defaults    env_keep += "ORACLE_HOME"
Defaults    env_keep += "LD_LIBRARY_PATH"
Defaults    env_keep += "TNS_ADMIN"
    这个问题就算解决了，之后就可以通过pip正常安装cx_Oracle了。
    问题2. libclntsh.so.10.1: cannot open shared object file: No such file or directory
    出现这个问题，是因为我一开始下载的instantclient的版本不是10g，而是11g，然后导致在ipython中执行import cx_Oracle的时候出现上述错误。
    这个主要是版本的问题，只要将instantclient版本改为10g就可以了。
    问题3. Pycharm引用时出现问题2的错误提示
    本来在ipython中执行import cx_Oracle已经没有问题了，但是在使用Pycharm执行Python项目的时候还是出现了问题2的错误。这个是因为LD_LIBARY_PATH的内容没能正常的引入到环境变量中，修改方法如下：
1
2
3
4
sudo vim /etc/ld.so.conf
# 然后将LD_LIBARY_PATH的内容添加到文件的最后，即下面一行内容
# /opt/oracle/instantclient_10_2
sudo ldconfig

'''
