import dropbox 

"""
HACKMIT 2015

Dropbox Photo Clustering


Tasks:
    authenticate user account
        Get App key and Secret key
        
    query folder of photos, given a folder name
    specify K
    represent photos as vectors/standardized format (different photo sizes, etc) (dropbox Api does this!)
    Function to parse input (image  vector  normalized vector)
    Have k-means/k-medoids methods up and running
    Pass in photos to k-means/k-mediods
    Have resulting k clusters, insert into k folders
    done. GG, self-driving car, OMW.
"""

fernando_code = "insert_code_here"
surya_code = "insert_code_here"
surya_image_folder = "sample_photos"
K = 10 # sample

def get_user_code():
    
    app_key = '71zx3gb8sya25eh'
    app_secret = '2t0fjsfqk7c55cx'
    
    flow = dropbox.client.DropboxOAuth2FlowNoRedirect(app_key, app_secret)
    
    authorize_url = flow.start()
    print 'Go to: ' + authorize_url
    code = raw_input("Enter the authorization code here: ").strip()
    return code
    
def authenticate(code):
    """
    Given a user acc code returns the client object of the user.
    
    Params:
        code: string obtained by running get_user_code(). Points to a specific 
              dropbox acc.
              
    Return:
        client: the object of the acc that the code points to.
    """
    
    app_key = 'app_key'
    app_secret = 'app_secret'
    flow = dropbox.client.DropboxOAuth2FlowNoRedirect(app_key, app_secret)
    print flow
    access_token, user_id = flow.finish(code)
    print access_token, user_id
    client = dropbox.client.DropboxClient(access_token)
    print 'linked account: ', client.account_info()
    return client

surya_code = get_user_code()
print surya_code 
authenticate(surya_code)

f = open('thoughts.txt', 'rb')
print f.read()

