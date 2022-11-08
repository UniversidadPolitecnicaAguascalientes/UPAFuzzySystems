#!/usr/bin/env python
# coding: utf-8


from skfuzzy import defuzz, trapmf, trimf, trimf, gaussmf, gbellmf
from numpy import array, arange, matrix, meshgrid, linspace, radians, fmin, fmax, max, zeros, minimum, maximum, isscalar, sum, prod
import matplotlib.pyplot as plt
from control import NonlinearIOSystem, sample_system, InterconnectedSystem, LinearIOSystem, tf2ss, tfdata


class fuzzy_universe():
    """Create a Universe Discrete or Continuous for Describe Inputs and Outputs in an 
    inference system.
    
    Args:
    name (str): String name for the universe.
    universe (ndarray): Numpy array with the elements of the universe.
    typeuniv (str): Specify with String `'continuous'`/`'discrete'` depending on the type of universe.

    Returns:
    U (obj): Object with the created universe.
    
    """

    def __init__(self,name='', universe=array([]), typeuniv ='continuous'):
        self.universe = universe
        self.name = name
        self.typeuniv = typeuniv
        self.structure =  {'name': self.name,'universe':self.universe}
    def add_fuzzyset(self, name_f, mf, vertices):

        """Allows to add a fuzzy set to the corresponding universe with name membership funtion
        and parameters.
       
        Args:
        name_f (str): String name for the fuzzy set
        mf (str): String with the membership funtion to use.
            `'raw'`: Directly specify the membership values in the vertices.
            `'trimf'`: Triangle membership function.
            `'trapmf'`: Trapezoid membership function.
            `'gaussmf'`: Gaussian membership function.
            `'gbellmf'`: Generalized bell membership function.
            `'eq'`: For specific equation to use in Takagi Sugeno forms, in this vertices is an string equation.
        vertices (list/str): List with the numerical parameters or string equation for the membership functions.
        
        Returns:
        none
        """

        self.structure[name_f] = [mf,vertices]
    def remove_fuzzyset(self,name_f=""):
        """Deletes from the universe the fuzzy set with the given name.
        
        Args:
        name_f (str): String name for the fuzzy set to delete.
        
        Returns: 
        none
        """
        del self.structure[name_f]
    def extract_fuzzyset(self,name):
        """ Extrats the universe and membership values of the fuzzy set in the universe passed with its name

        Args:
            name (str): Name of the fuzzy set in the universe.

        Returns:
            fuzzy_set (dict): Dictionary containing universe and membership values of the fuzzy set.
        """
        membershipvalues=[]
        X = self.structure['universe']
        membershipfuntion = self.structure[name][0]
        vertices = self.structure[name][1]
        exec('membershipvalues.append('+membershipfuntion+'(X,'+str(vertices)+'))')
        return {'universe':X,'membership values':membershipvalues[0]}
    def view_fuzzy(self):

        """Plots the the corresponding universe with all its fuzzy sets.
        

        Args:
        none

        Returns:
        none
        """

        fig = plt.subplots(figsize=(10,10))[1]
        colors = ['b','g','r','c','m','y','k']
        avy =0.08
        for i in self.structure:
            if i != 'universe' and i != 'name':
                func, vertex, universe, color = self.structure[i][0], self.structure[i][1],self.structure['universe'],colors.pop(-1)
                if len(colors)<1:
                    colors = ['b','g','r','c','m','y','k']
                if self.typeuniv == 'continuous':
                    if (func == 'trapmf'):
                        fig.plot(universe,trapmf(universe, vertex),color,label =i)
                    if (func == "trimf"):
                        fig.plot(universe,trimf(universe, vertex),color,label =i)
                    if (func == "gaussmf"):
                        fig.plot(universe,gaussmf(universe, vertex[0],vertex[1]),color,label =i)
                    if (func == "gbellmf"):
                        fig.plot(universe,gbellmf(universe, vertex[0],vertex[1],vertex[2]),color,label =i)
                    if (func == "eq"):
                        textv = 'x'
                        if textv not in vertex:
                            v = eval(vertex)
                            singleton = [1 if i==v else 0 for i in universe]
                            fig.stem(universe,singleton,color,color+'o',label =i+'='+vertex, use_line_collection = True)
                            plt.ylim([0, 1.1])

                        else:
                            avx = 0.02+len(vertex)*0.0015
                            singleton = [0.0001]*len(universe)
                            fig.stem(universe,singleton,color,color+'o',label =i, use_line_collection = True)
                            fig.text(-avx, avy, vertex, color=color,fontsize=24)
                            avy = avy-0.01
                            plt.xlim([-0.1,0.1])
                            plt.ylim([-0.1, 0.1])
                    if (func=="raw"):
                        fig.plot(universe,vertex,color,label =i)

                if self.typeuniv == 'discrete':
                    if (func == 'trapmf'):
                        fig.stem(universe,trapmf(universe, vertex),color,color+'o',label =i, use_line_collection = True)
                    if (func == "trimf"):
                        fig.stem(universe,trimf(universe, vertex),color,color+'o',label =i, use_line_collection = True)
                    if (func == "gaussmf"):
                        fig.stem(universe,gaussmf(universe, vertex[0],vertex[1]),color,color+'o',label =i, use_line_collection = True)
                    if (func == "gbellmf"):
                        fig.stem(universe,gbellmf(universe, vertex[0],vertex[1],vertex[2]),color,color+'o',label =i, use_line_collection = True)
                    if (func == "eq"):
                        textv = 'x'
                        if textv not in vertex:
                            v = eval(vertex)
                            singleton = [1 if i==v else 0 for i in universe]
                            fig.stem(universe,singleton,color,color+'o',label =i+'='+vertex, use_line_collection = True)
                            plt.ylim([0, 1.1])
                        else:
                            avx = 0.02+len(vertex)*0.0015
                            singleton = [0.0001]*len(universe)
                            fig.stem(universe,singleton,color,color+'o',label =i, use_line_collection = True)
                            fig.text(-avx, avy, vertex, color=color,fontsize=24)
                            avy = avy-0.01
                            plt.xlim([-0.1,0.1])
                            plt.ylim([-0.1, 0.1])
                    if (func=="raw"):
                        fig.stem(universe,vertex,color,color+'o',label =i, use_line_collection = True)

                        
        plt.title(self.structure['name'])
        plt.legend()
        plt.show()


class inference_system():
    """Allows to implement a fuzzy inference system with premises, consequences and rules. All universes of 
    premises and consequences must have the same number of elements.
    
    Args;
    name (str): String name for the created inference system.
    
    Returns:
    IS (obj): Object with the inference system created.
    """ 
    def __init__(self, name=''):
        self.name = name
        self.Defuzz = ''
        self.Implication = ''
        self.AndConector =  ''
        self.OrConector =  ''
        self.Agregattor = ''
        self.premises = []
        self.consequences = []
        self.rules = []
        self.structure = []
        self.output_universes = []
    def configure(self,typev=''):
        """Allows to configure the fuzzy inference system with a predefined structure.
    
        Args:
        typev (str): String name for determining the type of inference system to use.
            `'Linear'`: For linear inference systems.
            `'Sugeno'`: For Takagi-Sugeno systems.
            `'Mamdani'`: For Mamdani systems.
            `'FLSmidth'`: For FLSmidth systems base for controllers.
    
        Returns:
        none
        """ 

        if typev == 'Linear':
            self.Implication = 'prod'
            self.AndConector =  'prod'
            self.OrConector =  'probsum'
            self.Agregattor = 'probsum'
            self.Defuzz = 'centroidTS'
        if typev == 'Sugeno':
            self.Implication = 'prod'
            self.AndConector =  'prod'
            self.OrConector =  'probsum'
            self.Agregattor = 'probsum'
            self.Defuzz = 'centroidTS'
        if typev == 'Mamdani':
            self.Implication = 'min'
            self.AndConector =  'min'
            self.OrConector =  'max'
            self.Agregattor = 'max'
            self.Defuzz = 'centroid'
        if typev == 'FLSmidth':
            self.Implication = 'prod'
            self.AndConector =  'prod'
            self.OrConector =  'probsum'
            self.Agregattor = 'probsum'
            self.Defuzz = 'bisector'
    def add_premise(self,premise):
        """Allows to add a premise universe to the inference system.
        
        Args:
        premise (obj): Object with a fuzzy universe previously defined.

        Returns:
        none
        """
        self.premises.append(premise)
    def add_consequence(self,consequence):
        """Allows to add a consequence universe to the inference system.
        
        Args:
        consequence (obj): Object with a fuzzy universe previously defined.

        Returns:
        none
        """
        self.consequences.append(consequence)
        self.output_universes.append(consequence.universe)
    def add_rule(self,premises_r,conectors_r,consequence_r):
        """Allows to add a rule to the corresponding inference system with conectors in premises and consequences. All premises and consequences must have the same number of elements in their universes.
        
        Args:
        premises_r (list): List of lists of premises with two strings per premise [[`'Name Premise 1'`,`'Name Fuzzy Set 1'`],[`'Name Premise 2'`,`'Name Fuzzy Set 2'`]].
        conectors_r (list): List of conectors for the premises in the rule `'and'`/`'or'`.
        consequence_r (list): List of lists of consequences with two strings per consequence [[`'Name Consequence 1'`,`'Name Fuzzy Set 1'`],[`'Name Consequence 2'`,`'Name Fuzzy Set 2'`]].
        Returns:
        none
        """

        fuzzy_set = premises_r
        antecedents = []
        consequences_r = []
        universes = []
        universe = 0
        while fuzzy_set:
            premise = fuzzy_set.pop(0)
            for i in self.premises:
                if i.name == premise[0]:
                    provlist = i.structure[premise[1]].copy()
                    provlist.append(i.universe)
                    antecedents.append(provlist)
        while consequence_r:
            consequence = consequence_r.pop(0)
            for i in self.consequences:
                if i.name == consequence[0]:
                    consequences_r.append(i.structure[consequence[1]])
                    universes.append(universe)
                    universe = universe+1
        rule = {'antecedent':antecedents,
                  'conectors':conectors_r,
                  'consequent':consequences_r,
               'output_universes':universes}
        self.rules.append(rule)
    def build(self):
        """Built the created system with all the predefined configurations.
        
        Args:
        none

        Returns:
        none
        """
        
        self.structure = {'Rules': self.rules,
                'Defuzz': self.Defuzz,
                'Implication': self.Implication,
                'AndConector': self.AndConector,
                'OrConector': self.OrConector,
                'Agregattor': self.Agregattor,
                'output_universes': self.output_universes}

    def fuzzification(self,func,vertex,input_val,universe=array([])):
        """ Performs Fuzzification of the given intput with the funtion information, its parameters and the given universe.

        Args:
            func (str): define the name of the funtion to be used could be:
                `'raw'`: Directly specify the membership values in the vertices.
                `'trimf'`: Triangle membership function.
                `'trapmf'`: Trapezoid membership function.
                `'gaussmf'`: Gaussian membership function.
                `'gbellmf'`: Generalized bell membership function.
                `'eq'`: For specific equation to use in Takagi Sugeno forms, in this vertices is an string equation.
            vertex (list/str): List with the numerical parameters or string equation for the membership functions.
            input_val (float): single numerical value of the input in that universe to be fuzzified.
            universe (ndarray, optional): Universe for fuzzification, used only with the `'raw'` funtion.

        Returns:
            membership_value (ndarray): Single membership value for the single input value requested.
        """
        
        if isinstance(input_val, list):
            input_val = array(input_val).reshape(len(input_val),)
        if isscalar(input_val):
            input_val = [input_val]
            input_val = array(input_val).reshape(len(input_val),)
        if (func == 'trapmf'):
            membership=trapmf(input_val, vertex)
        if (func == "trimf"):
            membership=trimf(input_val, vertex)
        if (func == "gaussmf"):
            membership=gaussmf(input_val, vertex[0],vertex[1])
        if (func == "gbellmf"):
            membership=gbellmf(input_val, vertex[0],vertex[1],vertex[2])
        if (func == "eq"):
            x = input_val
            membership = eval(vertex)
        if (func == "raw"):
            v = array(vertex)
            membership = v[input_val==universe]
        return array(membership)
    def defuzzification(self,sc,universe,func):
        """ Defuzzifies the given fuzzy set with the method given.

        Args:
            sc (ndarray): Membership values of the output fuzzy set.
            universe (ndarray): Elements of the output universe.
            func (str): type of defuzzification:
                `'centroid'`: Centroid defuzzification.
                `'centroidTS'`: Centroid defuzzification for Takagi-Sugeno systems.
                `'bisector'`: Bisector of Area.
                `'mom'`: Mean of maxima.
                `'som'`: Smallest of maxima.
                `'lom'`: Largest of maxima.
        Returns:
            defuzzifiedv (float): Defuzzified value.
        """
        try:
            defuzzific = defuzz(universe,sc,func)
        except:
            defuzzific = 0
        return defuzzific

    def fuzzy_system_sim(self, inputs_val):
        """Simulate an input and returns the output of the defined inference system, rembember to check that all universes of premises and consequences have the same number of elements.

        Args:
            inputs_val (list): Input value per input premise in a list.

        Returns:
            output_val (ndarray): Output of the inference system.
        """
        IM = tuple([zeros((len(self.structure['Rules']),len(i))) for i in self.structure['output_universes']])
        miuo = zeros(len(self.structure['output_universes']))
        out_def = zeros(len(self.structure['output_universes']))
        for i in range(len(self.structure['Rules'])):
            conectorsv = self.structure['Rules'][i]['conectors'].copy()
            a = self.fuzzification(self.structure['Rules'][i]['antecedent'][0][0],self.structure['Rules'][i]['antecedent'][0][1],inputs_val[0],self.structure['Rules'][i]['antecedent'][0][2])
            for j in range(len(self.structure['Rules'][i]['antecedent'])-1):
                conect = conectorsv.pop(0)
                if conect =='and' and self.structure['AndConector']=='min':
                    a = minimum(a,self.fuzzification(self.structure['Rules'][i]['antecedent'][j+1][0],self.structure['Rules'][i]['antecedent'][j+1][1],inputs_val[j+1],self.structure['Rules'][i]['antecedent'][0][2]))
                if conect =='or' and self.structure['OrConector']=='max':
                    a = maximum(a,self.fuzzification(self.structure['Rules'][i]['antecedent'][j+1][0],self.structure['Rules'][i]['antecedent'][j+1][1],inputs_val[j+1],self.structure['Rules'][i]['antecedent'][0][2]))
                if conect =='and' and self.structure['AndConector']=='prod':
                    a = a*self.fuzzification(self.structure['Rules'][i]['antecedent'][j+1][0],self.structure['Rules'][i]['antecedent'][j+1][1],inputs_val[j+1],self.structure['Rules'][i]['antecedent'][0][2])
                if conect =='or' and self.structure['OrConector']=='probsum':
                    b = self.fuzzification(self.structure['Rules'][i]['antecedent'][j+1][0],self.structure['Rules'][i]['antecedent'][j+1][1],inputs_val[j+1],self.structure['Rules'][i]['antecedent'][0][2])
                    a = (a+b)-(a*b)
            for k in range(len(self.structure['Rules'][i]['consequent'])):
                if isinstance(self.structure['Rules'][i]['consequent'][k][0],str) and  self.structure['Defuzz'] != 'centroidTS':
                    self.structure['Rules'][i]['consequent'][k] = self.fuzzification(self.structure['Rules'][i]['consequent'][k][0],self.structure['Rules'][i]['consequent'][k][1],self.structure['output_universes'][self.structure['Rules'][i]['output_universes'][k]],self.structure['Rules'][i]['antecedent'][0][2])
                if self.structure['Defuzz'] != 'centroidTS':
                    if self.structure['Implication'] =='min':
                        IM[k][i,:] = fmin(a,self.structure['Rules'][i]['consequent'][k])
                    if self.structure['Implication'] =='prod':
                        IM[k][i,:] = a*self.structure['Rules'][i]['consequent'][k]
                else:
                    x = inputs_val
                    if self.structure['Implication'] =='min':
                        IM[k][i,0] = fmin(a,self.fuzzification(self.structure['Rules'][i]['consequent'][k][0],self.structure['Rules'][i]['consequent'][k][1],inputs_val,self.structure['Rules'][i]['antecedent'][0][2]))
                        miuo[k] = miuo[k] + a
                    if self.structure['Implication'] =='prod':
                        IM[k][i,0] = a*self.fuzzification(self.structure['Rules'][i]['consequent'][k][0],self.structure['Rules'][i]['consequent'][k][1],inputs_val,self.structure['Rules'][i]['antecedent'][0][2])
                        miuo[k] = miuo[k] + a
        for k in range(len(self.structure['output_universes'])):
            if self.structure['Defuzz'] != 'centroidTS':
                if self.structure['Agregattor'] =='max':
                    outf = max(IM[k],axis=0)
                if self.structure['Agregattor'] =='probsum':
                    outf = sum(IM[k],axis = 0)-prod(IM[k], axis= 0)
                out_def[k] = self.defuzzification(outf,self.structure['output_universes'][k],self.structure['Defuzz'])
            else:
                if miuo[k]>0:
                    out_def[k] = sum(IM[k][:,0])/miuo[k]
                else:
                    out_def[k] = 0
        return out_def

    def surface_fuzzy_system(self, inputvals=[],figsizeU=(15,15)):
        
        """Plots the response of the inference system for the list of input arrays for each premise, remember to check that premises and consequences universes have the same number of elements.
        

        Args:
        inputvals (list): List of numpy arrays containing the samples for each premise.

        Returns:
        none
        """
        
        surface_out = []
        if len(inputvals)==2:
            surface = array([[self.fuzzy_system_sim([i,j]) for j in inputvals[1]] for i in inputvals[0]]).reshape((len(inputvals[0]),len(inputvals[1])))
            X, Y = meshgrid(inputvals[0], inputvals[1])
            plt.figure(figsize=figsizeU)
            ax = plt.axes(projection='3d')
            print(X.shape)
            print(Y.shape)
            print(surface.shape)
            ax.plot_surface(X, Y, surface.transpose(), rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            ax.set_title('Surface Response: '+self.name)
            ax.set_xlabel(self.premises[0].name)
            ax.set_ylabel(self.premises[1].name)
            plt.show()
        if len(inputvals)==1:
            surface = array([self.fuzzy_system_sim([i]) for i in inputvals[0]]).reshape((len(inputvals[0])))
            plt.figure(figsize=figsizeU)
            ax = plt.axes()
            ax.plot(inputvals[0], surface)
            ax.set_title('Surface Response: '+self.name);
            ax.set_xlabel(self.premises[0].name)
            plt.show()

class fuzzy_controller():

    """Allows to implement a fuzzy controller system based on an inference system.
    

    Args:
    inference_system (obj): Predefined inference system with premises and consequences.
    typec (str): String to define the type of controller to use.
    `'PD-I'`: Proportional Derivative Integrative fuzzy controller.
    `'PD'`: Proportional Derivative fuzzy controller.
    `'P'`: Proportional fuzzy controller.
    `'Fuzzy1'`: Fuzzy controller with one input.
    `'Fuzzy2'`: Fuzzy controller with two inputs.
    `'tf'`: Transfer function of the process to control.
    `'DT'`: Sampling time to use in the controller.
    `'GE'`: Gains to use in some fuzzy controllers.
    `'GU'`: Gains to use in some fuzzy controllers.
    `'GCE'`: Gains to use in some fuzzy controllers.
    `'GIE'`: Gains to use in some fuzzy controllers.
    
    Returns:
    -----
    IS (obj): Object with the inference system created.
    """ 
    def __init__(self,inference_system,typec='',tf='',DT = 0.0001, GE=0.0, GU=0.0, GCE=0.0, GIE=0.0):
        self.DT = DT
        self.name = inference_system.name
        self.structure = inference_system.structure
        self.GE =  GE
        self.GU = GU
        self.GCE = GCE
        self.GIE = GIE
        self.typec = typec
        self.DT = DT
        self.controller = []
        self.tf =tf
        self.inference_system = inference_system
    def error_vals(self,t, x, u, params):
        error = u
        accumulated_error = x[0]+error
        previous_error = x[0]
        return [error,previous_error,accumulated_error]
    def error_val(self,t, x, u, params):
        error = [u[0]-u[1]]
        return error
    def fuzzy_PDIcontrol(self,t, x, u, params):
        fuzzy_controller, sampletime, GE, GU, GCE, GIE, error, previous_error, accumulated_error = params.get('fuzzy_controller'), params.get('DT'), params.get('GE'), params.get('GU'), params.get('GCE'), params.get('GIE'),x[0],x[1],x[2]
        derivative_error = (error-previous_error)/(sampletime)
        error_controller = error*GE
        derivative_controller = derivative_error*GCE
        integral_controller = accumulated_error*GIE
        control_out = (fuzzy_controller.fuzzy_system_sim([error_controller,derivative_controller])+integral_controller)*GU
        return control_out
    def fuzzy_PDcontrol(self,t, x, u, params):
        fuzzy_controller, sampletime, GE, GU, GCE, error, previous_error = params.get('fuzzy_controller'), params.get('DT'), params.get('GE'), params.get('GU'), params.get('GCE'), x[0], x[1]
        derivative_error = (error-previous_error)/(sampletime)
        error_controller = error*GE
        derivative_controller = derivative_error*GCE
        control_out = (fuzzy_controller.fuzzy_system_sim([error_controller,derivative_controller]))*GU
        return control_out
    def fuzzy_Pcontrol(self,t, x, u, params):
        fuzzy_controller, sampletime, GE, GU, error = params.get('fuzzy_controller'), params.get('DT'), params.get('GE'), params.get('GU'), u[0]
        error_controller = error*GE
        control_out = (fuzzy_controller.fuzzy_system_sim([error_controller]))*GU
        return control_out
    def fuzzy_control1(self,t, x, u, params):
        fuzzy_controller, error = params.get('fuzzy_controller'), u[0]
        error_controller = error
        control_out = (fuzzy_controller.fuzzy_system_sim([error_controller]))
        return control_out
    def fuzzy_control2(self,t, x, u, params):
        fuzzy_controller, sampletime, error,previous_error = params.get('fuzzy_controller'), params.get('DT'), x[0], x[1]
        derivative_error = (error-previous_error)/(sampletime)
        error_controller = error
        control_out = (fuzzy_controller.fuzzy_system_sim([error_controller,derivative_error]))
        return control_out
    def build(self):
        """Built the created controller with all the predefined configurations.
        
        Args:
        none

        Returns:
        none
        """
        if self.typec == 'PD-I':
            error_block = NonlinearIOSystem(
                None, self.error_val, name='error',
                inputs=2,
                outputs=1,
                dt = self.DT)
            TFd = sample_system(self.tf, self.DT, method='zoh')
            SS = tf2ss(TFd)
            Plant_block = LinearIOSystem(
                SS,
                inputs=1,
                outputs=1,
                states=len(tfdata(self.tf)[1][0][0])-1,
                name='plant')
            self.controller = NonlinearIOSystem(
                self.error_vals, self.fuzzy_PDIcontrol, name='FuzzyController',        # static system
                params = {'fuzzy_controller':self.inference_system, 'DT':self.DT,'GE':self.GE, 'GU': self.GU, 'GCE':self.GCE, 'GIE':self.GIE},
                inputs=1,
                outputs=1,
                states = 3,
                dt = self.DT)
            self.system = InterconnectedSystem(
                (error_block, self.controller, Plant_block),
                connections=(
                    ('error.u[1]', 'plant.y[0]'),
                    ('FuzzyController.u[0]', 'error.y[0]'),
                    ('plant.u[0]','FuzzyController.y[0]'),
                    ),
                inplist=('error.u[0]'),
                outlist=('plant.y[0]'),
                dt=self.DT)
        if self.typec == 'PD':
            error_block = NonlinearIOSystem(
                None, self.error_val, name='error',
                inputs=2,
                outputs=1,
                dt = self.DT)
            TFd = sample_system(self.tf, self.DT, method='zoh')
            SS = tf2ss(TFd)
            Plant_block = LinearIOSystem(
                SS,
                inputs=1,
                outputs=1,
                states=len(tfdata(self.tf)[1][0][0])-1,
                name='plant')
            self.controller = NonlinearIOSystem(
                self.error_vals, self.fuzzy_PDcontrol, name='FuzzyController',        # static system
                params = {'fuzzy_controller':self.inference_system, 'DT':self.DT,'GE':self.GE, 'GU': self.GU, 'GCE':self.GCE},
                inputs=1,
                outputs=1,
                states = 3,
                dt = self.DT)
            self.system = InterconnectedSystem(
                (error_block, self.controller, Plant_block),
                connections=(
                    ('error.u[1]', 'plant.y[0]'),
                    ('FuzzyController.u[0]', 'error.y[0]'),
                    ('plant.u[0]','FuzzyController.y[0]'),
                    ),
                inplist=('error.u[0]'),
                outlist=('plant.y[0]'),
                dt=self.DT)
        if self.typec == 'P':
            error_block = NonlinearIOSystem(
                None, self.error_val, name='error',
                inputs=2,
                outputs=1,
                dt = self.DT)
            TFd = sample_system(self.tf, self.DT, method='zoh')
            SS = tf2ss(TFd)
            Plant_block = LinearIOSystem(
                SS,
                inputs=1,
                outputs=1,
                states=len(tfdata(self.tf)[1][0][0])-1,
                name='plant')
            self.controller = NonlinearIOSystem(
                None, self.fuzzy_Pcontrol, name='FuzzyController',        # static system
                params = {'fuzzy_controller':self.inference_system, 'DT':self.DT,'GE':self.GE, 'GU': self.GU},
                inputs=1,    # system inputs
                outputs=1,                            # system outputs
                dt = self.DT)
            self.system = InterconnectedSystem(
                (error_block, self.controller, Plant_block),
                connections=(
                    ('error.u[1]', 'plant.y[0]'),
                    ('FuzzyController.u[0]', 'error.y[0]'),
                    ('plant.u[0]','FuzzyController.y[0]'),
                    ),
                inplist=('error.u[0]'),
                outlist=('plant.y[0]'),
                dt=self.DT)
        if self.typec == 'Fuzzy1':
            error_block = NonlinearIOSystem(
                None, self.error_val, name='error',
                inputs=2,
                outputs=1,
                dt = self.DT)
            TFd = sample_system(self.tf, self.DT, method='zoh')
            SS = tf2ss(TFd)
            Plant_block = LinearIOSystem(
                SS,
                inputs=1,
                outputs=1,
                states=len(tfdata(self.tf)[1][0][0])-1, 
                name='plant')
            self.controller = NonlinearIOSystem(
                None, self.fuzzy_control1, name='FuzzyController',        # static system
                params = {'fuzzy_controller':self.inference_system, 'DT':self.DT},
                inputs=1,    # system inputs
                outputs=1,                            # system outputs
                dt = self.DT)
            self.system = InterconnectedSystem(
                (error_block, self.controller, Plant_block),
                connections=(
                    ('error.u[1]', 'plant.y[0]'),
                    ('FuzzyController.u[0]', 'error.y[0]'),
                    ('plant.u[0]','FuzzyController.y[0]'),
                    ),
                inplist=('error.u[0]'),
                outlist=('plant.y[0]'),
                dt=self.DT)
        if self.typec == 'Fuzzy2':
            error_block = NonlinearIOSystem(
                None, self.error_val, name='error',
                inputs=2,
                outputs=1,
                dt = self.DT)
            TFd = sample_system(self.tf, self.DT, method='zoh')
            SS = tf2ss(TFd)
            Plant_block = LinearIOSystem(
                SS,
                inputs=1,
                outputs=1,
                states=len(tfdata(self.tf)[1][0][0])-1,
                name='plant')
            self.controller = NonlinearIOSystem(
                self.error_vals, self.fuzzy_control2, name='FuzzyController',        # static system
                params = {'fuzzy_controller':self.inference_system, 'DT':self.DT},
                inputs=1,
                outputs=1,
                states = 3,
                dt = self.DT)
            self.system = InterconnectedSystem(
                (error_block, self.controller, Plant_block),
                connections=(
                    ('error.u[1]', 'plant.y[0]'),
                    ('FuzzyController.u[0]', 'error.y[0]'),
                    ('plant.u[0]','FuzzyController.y[0]'),
                    ),
                inplist=('error.u[0]'),
                outlist=('plant.y[0]'),
                dt=self.DT)
    def get_controller(self,showinfo=False):
        """Returns the block with the created fuzzy controller for simulation using `control` liblrary.
        
        Args:
        showinfo (bool): Show extra information for the controller.

        Returns:
        none
        """
        if self.typec == 'PD-I' and showinfo:
            print(self.controller)
        if self.typec == 'PD' and showinfo:
            print(self.controller)
        if self.typec == 'P' and showinfo:
            print(self.controller)
        return self.controller
    def get_system(self,showinfo=False):
        """Returns the block with the created fuzzy controller and the transfer function of the process for simulation using `control` liblrary.
        
        Args:
        showinfo (bool): Show extra information for the controller.

        Returns:
        none
        """
        if showinfo:
            print(self.system)
        return self.system
