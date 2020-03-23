"""
This file contains a simple mysql database connector and a child SegmentationDatabaseConnector class 

Author: Konstantin Ditschuneit
Date: 19.08.2019
"""
import mysql.connector  # connection to mysql
from mysql.connector import Error  # catch query errors
import os  # paths
from shutil import copyfile # copy files
import img.compare as comp #create hashs
import cv2 # image functions
from database.segmentation_image import SegmentationImage # Segmentation image
import threading # lock 


class DatabaseConnector:
    """
    DatabaseConnector is used to perform queries to a mysql database

    Attributes
    ----------
    host: str
        adress of database
    database: str
        name of database
    user: str
        database user
    password: str
        password of database user 
    connection: mysql.connection
        variable storing a connection to the database
    lock: threading.lock
        lock used to ensure, that only a single connection is established

    """

    def __init__(self):
        self.connection = None
        self.lock = threading.Lock()
        self.host = 'localhost'
        self.database = 'kitcar'
        self.user = 'kitcar'
        self.password = 'kitcar'

    def get_connection(self):
        self.lock.acquire()
        try:

            if self.connection is None:
                self.connection = mysql.connector.connect(host=self.host,
                                                          database=self.database,
                                                          user=self.user,
                                                          password=self.password)
            if self.connection.is_connected():
                print("Connected as ", self.user,
                      "to MySQL database ", self.database)
                return self.connection
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            self.lock.release()

        return None

    def get_cursor(self):
        """
        returns a cursor, which can be used for queries to database
            None, if the connection couldnt be opened
        """
        con = self.get_connection()

        # Return none if connection failed
        if con == None:
            return None

        return con.cursor()

    def close(self):
        """
        close connection to database
        """
        if(self.connection.is_connected()):
            self.connection.close()
            print("MySQL connection is closed")

    def create_insert_query(self, table, params):
        """
        Returns an insertion query as a string

        Attributes
        ----------
        table: str
            name of table to be queried from

        params: dictionary {str:any} 
            keys and values in query

        Return
        ----------
        sql,values (str,array):
            tuple of the mysql syntax insertion string and all values in an array

        """
        key_str = "("  # string which will contain the keys
        # string (%s,%s,...) which will contain one %s for each key
        perc_str = "("
        vals = []  # list which will be filled with all values

        for key in params:
            key_str += "`" + key + "`,"
            perc_str += "%s,"
            vals.append(params[key])

        key_str = key_str.rstrip(',') + ")"
        perc_str = perc_str.rstrip(',') + ")"

        sql = "INSERT INTO " + table + " " + key_str + \
            " VALUES " + perc_str  # combine to string

        return sql, vals

    def execute(self, query, param_list=None):
        """
        Executes the given query

        Attributes
        ----------
        query: str
            query in mysql syntax

        params_list: list
            list, which contains all values that should be passed as arguments into the query

        Return
        ----------
        result:
            returns what the cursor gives back (e.g. list from selection query)


        """
        # Get cursor
        cursor = self.get_cursor()  # Get/create cursor

        cursor.execute(query, param_list)  # execute the query

        print(query)
        res = None
        try:
            res = cursor.fetchall()  # Attempt to get results of the query
        except Error as e:
            print("Error while executing query: ", e)

        cursor.close()  # Close cursor

        self.connection.commit()  # Commit to connection

        return res

class SegmentationDatabaseConnector(DatabaseConnector):
    """
    SegmentationDatabaseConnector child of DatabaseConnector to perform specific tasks for loading segmentation data

    Attributes
    ----------
    table: str
        name of database table
    input_folder:
        folder where input images in database are stored
    mask_folder 
        Where masks in database are stored

    """

    def __init__(self, table='segmentation_image', 
                        input_folder=os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data','database','inputs'), 
                        mask_folder=os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data','database','masks')):
        self.table = table
        self.input_folder = input_folder
        self.mask_folder = mask_folder
        
        super(SegmentationDatabaseConnector, self).__init__()

    def insert(self, img_path, mask_path=None, dataset_name=None):
        """
        Insert image at img_path into database

        Parameters
        ----------
        img_path: str
            path of image to be saved into database
        mask_path: str, optional
            path of mask to be saved into database
        """

        img = cv2.imread(img_path)  # load image

        small_bin = comp.small_bin(img)  #create small binarized string of image
        hash_str = comp.hash_image(img) # create hash of image

        params = {"key": hash_str} # parameters 
        params['small_binary'] = small_bin 

        if not dataset_name is None:
            params['dataset_name'] = dataset_name

        try:
            print('Attempting to insert image ' + hash_str)

            self.execute(*self.create_insert_query(self.table, params)) # Run query
        except mysql.connector.IntegrityError as e:
            if e.errno == 1062:
                print('DUPLICATE!')
            pass

        # Save image to db folder
        new_path = os.path.join(self.input_folder, hash_str + '.png') 

        os.makedirs(self.input_folder, exist_ok=True) # Create folders if they dont exist yet

        copyfile(img_path, new_path)

        if not mask_path is None:
            new_mask_path = os.path.join( self.mask_folder, hash_str + '.png')
            os.makedirs(self.mask_folder, exist_ok=True)
            copyfile(mask_path, new_mask_path)

        return hash_str

    def insert_folder(self, img_folder, mask_folder=None, dataset_name=None, delete=False):
        """
        Insert all images of folder into database

        Parameters
        ----------
        img_folder: str
            path of folder with images to be saved into database
        mask_folder: str, optional
            path of folder with masks to be saved into database
        delete: bool
            if images should be deleted from img_folder once added to database
        """
        for file in os.listdir(img_folder): # loop through all images in folder
            img_path = os.path.join(img_folder, file) 

            if mask_folder is None:
                self.insert(img_path, None, dataset_name=dataset_name) #Insert only image
            else:
                mask_path = os.path.join(mask_folder, file)
                self.insert(img_path, mask_path, dataset_name=dataset_name) # insert image and mask

            if delete:
                os.remove(img_path) # delete image

                if not mask_folder is None:
                    os.remove(mask_path) #delete mask

    def load_keys(self, dataset_name=None, random=False, max_count=0):
        """
        Load all keys from in database

        Parameters
        ----------
        random: bool, optional
            keys are given back randomly
        max_count: int, optional
            maximal number of keys that are returned
        """
        query = "SELECT `key` FROM " + self.table # query

        if not dataset_name is None:
            query += " WHERE `dataset_name`='" + dataset_name + "'" # search query
        if random == True:
            query += " ORDER BY RAND()" #if random
        if not max_count == 0:
            query += " LIMIT " + str(max_count) #set limit

        res = self.execute(query) #execute query

        return list(map(lambda x: x[0], res)) #filter keys

    def load(self, key):
        """
        Load segmentation image with key

        Parameters
        ----------
        key: str
            key of image
        """
        query = "SELECT * FROM " + self.table + " WHERE `key`='" + key + "'" # search query
        print(query)
        res = self.execute(query) 

        if len(res) > 0:
            return SegmentationImage(arg= res[0], input_folder=self.input_folder, mask_folder= self.mask_folder) # create SegmentationImage

        return None

    def load_all(self, dataset_name=None, random=False, max_count=0):
        """
        Load all segmentation images from database

        Parameters
        ----------
        random: bool, optional
            images are given back randomly
        max_count: int, optional
            maximal number of images that are returned
        """
        # Perform search query
        query = "SELECT * FROM " + self.table 
        print(query)

        if not dataset_name is None:
            query += " WHERE `dataset_name`='" + dataset_name + "'" # search query
        if random == True:
            query += " ORDER BY RAND()"
        if not max_count == 0:
            query += " LIMIT " + str(max_count)

        res = self.execute(query)

        imgs = []
        for r in res:
            imgs.append(SegmentationImage(arg= r, input_folder=self.input_folder, mask_folder= self.mask_folder))

        return imgs
