import os 

def duplicate_path(path):
    duplicated_path = path
    n = 0
    while os.path.exists(duplicated_path):
        n += 1
        duplicated_path = path + '_' + str(n)
    os.makedirs(duplicated_path)
    return duplicated_path

# Deleted by Soonbeom and use create_path()
'''
def configsetup_path(path, config):
    configsetup_path = path
    #n = 0
    #while os.path.exists(duplicated_path):
    #    n += 1
    #    duplicated_path = path + '_' + str(n)
    configsetup_path = configsetup_path + 'mellen' + str(config.mel_length) + '_textembedsize' + str(config.tts_text_embed_size) + '_noteembedsize' + str(config.tts_note_embed_size)
    if os.path.exists(configsetup_path):
        return configsetup_path
    else:
        os.makedirs(configsetup_path)
    return configsetup_path

def create_configsetuppath(path, config, action="configsetup", verbose=True):
    created_path = path
    verbose_message = 'created'
    if os.path.exists(path):
        if action == 'configsetup':
            created_path = configsetup_path(path, config)
        elif action == 'error':
            raise AssertionError("\'%s\' already exists." % path)
        else: 
            raise AssertionError("Invalid action is used.")
    else:
        os.makedirs(path)

    if verbose:
        print("\'%s\' is %s." % (created_path, verbose_message))

    return created_path
'''

def create_path(path, action="duplicate", verbose=True):
    created_path = path
    verbose_message = 'created'
    if os.path.exists(path):
        if action == 'overwrite':
            os.makedirs(path, exist_ok=True)
            verbose_message = 'overwritten'
        elif action == 'duplicate':
            created_path = duplicate_path(path)
        elif action == 'error':
            raise AssertionError("\'%s\' already exists." % path)
        else: 
            raise AssertionError("Invalid action is used.")
    else:
        os.makedirs(path)

    if verbose:
        print("\'%s\' is %s." % (created_path, verbose_message))

    return created_path

class FileXT(object):
    def __init__(self, *args):
        path_list = []
        self.ext = ''
        for path in args:
            for p in path.split('/'):
                if p == '.':
                    p = os.getcwd()
                    path_list.append(p)
                elif p == '..':
                    p = os.path.dirname(os.getcwd())
                    path_list.append(p)
                elif len(p) > 0 and p[0] == '.':
                    self.ext = p
                else: 
                    if '.' in p:
                        self.ext = '.' + p.split('.')[-1]
                        p = p.split('.')[0]
                    
                    path_list.append(p)

        filestem = '/'.join(path_list)
        self.basename = os.path.basename(filestem) + self.ext
        self.basestem = os.path.basename(filestem) 
        self.filepath = os.path.dirname(filestem)
        self.filename = filestem + self.ext
        self.filestem = filestem