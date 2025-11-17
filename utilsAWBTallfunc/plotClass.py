import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class Plot_maker(object):
    def __init__(self, FF, legend_list, label_list, pic_name):
        self.FF = FF
        self.legend_list = legend_list
        self.label_list = label_list
        self.pic_name = pic_name
        self.ylim = [0.01,2.0]
        self.xlabel = "Training steps"
        self.legend_location = 'lower left'
        self.dpi = 600
        self.fig_size = (12,6)
        self.legend_coor = (-.01,-.04)
        self.color_list = ['salmon', 'cornflowerblue', 'mediumseagreen','darkorange','#8C67AF','#8D57C1', '#704699','#4D0A6A']
        self.title = "Comparing Proposed Training Method to Standard Training"
        self.log_scale = True

    def load_data(self):
        Data=[]
        for files in self.FF:
            TT = []
            VV= []
            TeTe= []
            for file in files:    
                with open(file, 'rb') as f:
                    loaded_dict = pickle.load(f)
                Train = []
                Valid = []
                Test =  []
                #print(loaded_dict)        
                for key in loaded_dict.keys():
                    dic = loaded_dict[key]
                    train = []
                    valid = []
                    test =  []
                    #print(len(dic.keys()), key)
                    for keys in dic.keys():
                        # print( np.array(dic[keys]).shape )
                        if keys.startswith('train'):
                            train.append( dic[keys])
                        if keys.startswith('valid'):
                            valid.append( dic[keys])
                        if keys.startswith('test'):
                            test.append( dic[keys])
                    Test.append(np.array(test))
                    Valid.append(np.array(valid))
                    Train.append(np.array(train))
                TT.append(np.array(Train))
                VV.append(np.array(Valid))
                TeTe.append(np.array(Test))
            Data.append( (TT, VV, TeTe) )
        with open('processed_data.pkl', 'wb') as f:
            pickle.dump(Data, f)
        with open("processed_data.pkl", 'rb') as f:
            Data = pickle.load(f)
        return Data
    
    def make_plot(self, mode = 'train'):
        Data = self.load_data()
        #print(Data)
        CB91_Blue = '#05E1E1'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        CB91_Grad_BP = ['#2CBDFE', '#2FB9FC', '#33B4FA', '#36B0F8',
                        '#3AACF6', '#3DA8F4', '#41A3F2', '#449FF0',
                        '#489BEE', '#4B97EC', '#4F92EA', '#528EE8',
                        '#568AE6', '#5986E4', '#5C81E2', '#607DE0',
                        '#6379DE', '#6775DC', '#6A70DA', '#6E6CD8',
                        '#7168D7', '#7564D5', '#785FD3', '#7C5BD1',
                        '#7F57CF', '#8353CD', '#864ECB', '#894AC9',
                        '#8D46C7', '#9042C5', '#943DC3', '#9739C1',
                        '#9B35BF', '#9E31BD', '#A22CBB', '#A528B9',
                        '#A924B7', '#AC20B5', '#B01BB3', '#B317B1']
        small = 14
        med = 14
        large = 16
        plt.style.use("seaborn-v0_8-deep")
        COLOR = 'dimgrey'
        TCOLOR = 'darkslategrey'
        rc={'axes.titlesize': small,
            'legend.fontsize': small,
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small,
            'font.family': "sans-serif",
            'font.sans-serif': "Arial",
            'text.color' : TCOLOR,
            'axes.labelcolor' : TCOLOR,
            'axes.axisbelow': False,
            'axes.edgecolor': COLOR,
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': TCOLOR,
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': TCOLOR,
            'xtick.bottom': True,
            'xtick.color': TCOLOR,
            'xtick.direction': 'in',
            'xtick.top': False,
            'ytick.color': TCOLOR,
            'ytick.direction': 'in',
            'ytick.left': True,
            'ytick.right': False}
        plt.rcParams.update(rc)
        plt.rc('text', usetex = False)
        offset=3
        def letter_annotation(ax, xoffset, yoffset, letter):
            ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=20, weight='bold')
        fig = plt.figure(figsize=self.fig_size, dpi =self.dpi)
        plt.suptitle(self.title)
        (col1fig)= fig.subfigures(1, 1, width_ratios=[1])
        col1_axs = col1fig.subplots(1,1, sharex=True)
        # col1fig.subplots_adjust(wspace=0.2)
        ax = col1_axs
        
        for k in range(len(Data)):
            (TT, VV, TeTe) = Data[k]
            listum =[0,5]
            steps = np.arange(TT[0].shape[1])
            Train = TT[0]
            Validate = VV[0]
            Test = TeTe[0]
            steps1 = np.arange(TeTe[0].shape[1])
            steps1 = steps1*500+500
            color = self.color_list[k]

            steps = np.arange(TT[0].shape[1])
            Train = TT[0]
            steps1 = np.arange(TeTe[0].shape[1])
            Test = TeTe[0]
            color = self.color_list[k]
            #print(TeTe)
            testlist = []
            steps1 = steps1*500+500
            #steps1 = steps1*250+250

            steps2 = np.arange(VV[0].shape[1])
            #steps2 = steps2*100+100
            steps2 = steps2*50+50
            #--new
            if mode == 'train':
                for j, label in zip([0], self.label_list):
                    mu = np.mean(Train[:,:,listum[j] ], axis=0)
                    var= np.std(Train[:,:,listum[j]], axis=0)
                    ax.plot(steps, mu, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k])
                    ax.fill_between(steps, mu-(var/2), mu+(var/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
            elif mode == 'test':
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Test[:,:,listum[j] ], axis=0)
                    var1= np.std(Test[:,:,listum[j]], axis=0)
                    ax.plot(steps1, mu1, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k])
                    ax.set_ylabel(label)
            elif mode == 'validate':
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Validate[:,:,listum[j] ], axis=0)
                    var1= np.std(Validate[:,:,listum[j]], axis=0)
                    ax.plot(steps2, mu1, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k])
                    ax.set_ylabel(label)
            elif mode == 'train and validate':
                for j, label in zip([0], self.label_list):
                    mu = np.mean(Train[:,:,listum[j] ], axis=0)
                    var= np.std(Train[:,:,listum[j]], axis=0)
                    ax.plot(steps, mu, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k] + ' (train)')
                    ax.fill_between(steps, mu-(var/2), mu+(var/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Validate[:,:,listum[j] ], axis=0)
                    var1= np.std(Validate[:,:,listum[j]], axis=0)
                    
                    ax.plot(steps2, mu1, color = color, linewidth=0.5, linestyle='--', label=self.legend_list[k] + ' (validate)')
                    ax.fill_between(steps2, mu1-(var1/2), mu1+(var1/2),interpolate=True, alpha=0.2, color=color)
            elif mode == 'test and validate':
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Test[:,:,listum[j] ], axis=0)
                    var1= np.std(Test[:,:,listum[j]], axis=0)
                    
                    ax.plot(steps1, mu1, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k] + ' (test)')
                    ax.fill_between(steps1, mu1-(var1/2), mu1+(var1/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
                for j, label in zip([0], self.label_list):
                    mu2 = np.mean(Validate[:,:,listum[j] ], axis=0)
                    var2= np.std(Validate[:,:,listum[j]], axis=0)
                    ax.plot(steps2, mu2, color = color, linewidth=0.5, linestyle='--', label=self.legend_list[k] + ' (validate)')
                    ax.fill_between(steps2, mu2-(var2/2), mu2+(var2/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
            elif mode == 'train and test':
                for j, label in zip([0], self.label_list):
                    mu = np.mean(Train[:,:,listum[j] ], axis=0)
                    var= np.std(Train[:,:,listum[j]], axis=0)
                    ax.plot(steps, mu, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k] + ' (train)')
                    ax.fill_between(steps, mu-(var/2), mu+(var/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Test[:,:,listum[j] ], axis=0)
                    var1= np.std(Test[:,:,listum[j]], axis=0)
                    
                    ax.plot(steps1, mu1, color = color, linewidth=0.5, linestyle='--', label=self.legend_list[k] + ' (test)')
                    ax.fill_between(steps1, mu1-(var1/2), mu1+(var1/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
            else:
                for j, label in zip([0], self.label_list):
                    mu = np.mean(Train[:,:,listum[j] ], axis=0)
                    var= np.std(Train[:,:,listum[j]], axis=0)
                    ax.plot(steps, mu, color = color, linewidth=0.5, linestyle='-', label=self.legend_list[k] + ' (train)')
                    ax.fill_between(steps, mu-(var/2), mu+(var/2),interpolate=True, alpha=0.2, color=color)
                    ax.set_ylabel(label)
                for j, label in zip([0], self.label_list):
                    mu1 = np.mean(Test[:,:,listum[j] ], axis=0)
                    var1= np.std(Test[:,:,listum[j]], axis=0)
                    ax.plot(steps1, mu1, color = color, linewidth=0.5, linestyle='--', label=self.legend_list[k] + ' (test)')
                    ax.set_ylabel(label)
                for j, label in zip([0], self.label_list):
                    mu2 = np.mean(Validate[:,:,listum[j] ], axis=0)
                    var2= np.std(Validate[:,:,listum[j]], axis=0)
                    ax.plot(steps2, mu2, color = color, linewidth=0.5, linestyle='--', label=self.legend_list[k] + ' (validate)')
                    ax.set_ylabel(label)
            #----
            ax.set_ylim(self.ylim)
            # ax[1].set_ylim([0,5])
            if self.log_scale == True:
                ax.set_yscale('log')
            ax.set_xlabel(self.xlabel)
            # ax[1].set_xlabel('training steps')
        # letter_annotation(ax[0], -.27, 1.1, 'B')
        plt.legend(loc=self.legend_location, bbox_to_anchor=self.legend_coor, ncol=1)
        sns.despine(offset=5, trim=False)
        #plt.show()
        plt.savefig(self.pic_name + '.png', dpi=self.dpi)
        
        