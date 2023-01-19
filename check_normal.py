from multiprocessing import Process
import os

set1 = set()
set2 = set()
set3 = set()
set4 = set()

class CheckingNorm(Process):
    def __init__(self):
        Process.__init__(self)

    def run_norm(self, o1, o2, o3, o4):
        set1 = set(o1.qtolist())  # norm1)
        set2 = set(o2.qtolist())  # norm2)
        set3 = set(o3.qtolist())  # norm3)
        set4 = set(o4.qtolist())  # norm4)
        '''
        print('norm1 size: ', o1.qsize())  # norm1)
        print('norm2 size: ', o2.qsize())  # norm2)
        print('norm3 size: ', o3.qsize())  # norm3)
        print('norm4 size: ', o4.qsize())  # norm4)

        print('set1: ', set1)
        print('set2: ', set2)
        print('set3: ', set3)
        print('set4: ', set4)
        '''
        norm = set1 & set2 & set3 & set4

        if norm:
            print('norm is ', norm)
            return norm
        else:
            norm = None
            return norm

def check_norm(**kwargs):

    '''
    while not kwargs['stop_event'].wait(1e-9):
        output1 = kwargs['que_out_1']
        output2 = kwargs['que_out_2']
        output3 = kwargs['que_out_3']
        output4 = kwargs['que_out_4']
        print('output1: ', output1.qtolist())
        print('output2: ', output2.qtolist())
        print('output3: ', output3.qtolist())
        print('output4: ', output4.qtolist())
    '''

    output1 = kwargs['que_out_1']
    output2 = kwargs['que_out_2']
    output3 = kwargs['que_out_3']
    output4 = kwargs['que_out_4']

    while not kwargs['stop_event'].wait(1e-9):
        if output1.qsize() > 0 and output2.qsize() > 0 and output3.qsize() > 0 and output4.qsize() > 0:          

            rm_output = kwargs['check_norm'].run_norm(output1, output2, output3, output4)

            if rm_output:
                print('there are noraml!')
                for n in rm_output:
                    os.rename(n, n.split('.')[0] + '_0000.png')
                    output1.remove(n)
                    output2.remove(n)
                    output3.remove(n)
                    output4.remove(n)
                '''
                print('after norm1 size: ', output1.qsize())  # norm1)
                print('after norm2 size: ', output2.qsize())  # norm2)
                print('after norm3 size: ', output3.qsize())  # norm3)
                print('after norm4 size: ', output4.qsize())  # norm4)
                '''
            # else: print('there are not intersection!')
    
            
