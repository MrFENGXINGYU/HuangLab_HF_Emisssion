import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed


KEY_FEATURES = ['X64', 'X56', 'X103', 'X105', 'X106']

BOUNDS = {
    'X64': (767, 830),
    'X56': (875, 910),
    'X103': (202, 212),
    'X105': (575, 639),
    'X106': (184, 189)
}


def evaluate(z, x_i, y_i, model, bounds):
    x_new = x_i.copy()
    for j, feat in enumerate(KEY_FEATURES):
        val = z[j]
        l, u = bounds[feat]
        if val < l:
            val = l + np.random.uniform(0, (u - l) * 0.05)
        elif val > u:
            val = u - np.random.uniform(0, (u - l) * 0.05)
        x_new[feat] = val
    
    x_new = x_new.reindex(x_i.index.tolist())
    pred_y = model.predict(x_new.values.reshape(1, -1))[0]
    violation = max(0, pred_y - y_i)
    return (pred_y, violation)


def adjust_params(pop, cxpb_init, mutpb_init):
    vals = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
    if not vals:
        return cxpb_init, mutpb_init
    
    std = np.std(vals)
    cxpb = min(cxpb_init + 0.1, 0.9) if std < 10 else cxpb_init
    mutpb = min(mutpb_init + 0.1, 0.5) if std < 10 else mutpb_init
    return cxpb, mutpb


def check_bounds(ind, bounds):
    for j, feat in enumerate(KEY_FEATURES):
        l, u = bounds[feat]
        if ind[j] < l:
            ind[j] = l + np.random.uniform(0, (u - l) * 0.05)
        elif ind[j] > u:
            ind[j] = u - np.random.uniform(0, (u - l) * 0.05)
    return ind


def optimize_sample(idx, x_i, y_i, model, bounds):
    if "FitnessMulti" not in creator.__dict__:
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    for feat in KEY_FEATURES:
        toolbox.register(f"attr_{feat}", np.random.uniform, bounds[feat][0], bounds[feat][1])
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [toolbox.__getattribute__(f"attr_{feat}") for feat in KEY_FEATURES], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, x_i=x_i, y_i=y_i, model=model, bounds=bounds)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20,
                    low=[b[0] for b in bounds.values()], up=[b[1] for b in bounds.values()])
    toolbox.register("mutate", tools.mutGaussian, mu=0,
                    sigma=[(b[1] - b[0]) * 0.1 for b in bounds.values()], indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    
    pop = toolbox.population(n=200)
    cxpb, mutpb = 0.6, 0.3
    best_fit = (float('inf'), float('inf'))
    no_improve = 0
    
    for gen in range(300):
        if gen == 0:
            fits = list(map(toolbox.evaluate, pop))
            for fit, ind in zip(fits, pop):
                ind.fitness.values = fit
        
        cxpb, mutpb = adjust_params(pop, 0.6, 0.3)
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        for ind in offspring:
            ind[:] = check_bounds(ind, bounds)
        
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, k=200)
        
        curr_best = min(pop, key=lambda x: (x.fitness.values[1], x.fitness.values[0]))
        if (abs(curr_best.fitness.values[0] - best_fit[0]) < 1e-5 and
            abs(curr_best.fitness.values[1] - best_fit[1]) < 1e-5):
            no_improve += 1
        else:
            no_improve = 0
            best_fit = curr_best.fitness.values
        
        if no_improve >= 50:
            break
    
    z_best = min(pop, key=lambda x: (x.fitness.values[1], x.fitness.values[0]))
    z_dict = dict(zip(KEY_FEATURES, z_best))
    
    x_new = x_i.copy()
    x_new.update(z_dict)
    x_new = x_new.reindex(x_i.index.tolist())
    y_pred = model.predict(x_new.values.reshape(1, -1))[0]
    
    result = {'idx': idx, 'y_orig': y_i, 'y_opt': y_pred}
    for k in KEY_FEATURES:
        result[f'orig_{k}'] = x_i[k]
        result[f'opt_{k}'] = z_dict[k]
    
    return result


def run_optimization(X, y, model, bounds=BOUNDS, output='results.csv'):
    tasks = [(i, X.loc[i], y.loc[i], model, bounds) for i in X.index]
    
    results = []
    for task in tqdm(tasks, desc="Optimizing"):
        res = optimize_sample(*task)
        results.append(res)
    
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)
    
    success = (df['y_opt'] < df['y_orig']).sum()
    print(f"\nSuccess: {success}/{len(df)} ({success/len(df)*100:.1f}%)")
    print(f"Avg reduction: {(df['y_orig'] - df['y_opt']).mean():.4f}")
    
    return df


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    model = joblib.load('model.pkl')
    
    feature_cols = ['X30', 'X59', 'X58', 'X57', 'X56', 'X55', 'X54', 'X51', 'X48', 'X47', 
                   'X46', 'X45', 'X60', 'X44', 'X42', 'X41', 'X40', 'X39', 'X38', 'X37', 
                   'X36', 'X35', 'X34', 'X32', 'X91', 'X43', 'X62', 'X63', 'X64', 'X90', 
                   'X9', 'X89', 'X88', 'X87', 'X86', 'X85', 'X84', 'X83', 'X82', 'X81', 
                   'X80', 'X79', 'X78', 'X77', 'X76', 'X75', 'X71', 'X70', 'X7', 'X69', 
                   'X68', 'X67', 'X66', 'X65', 'X3', 'X27', 'X92', 'X25', 'X129', 'X128', 
                   'X26', 'X126', 'X125', 'X124', 'X122', 'X121', 'X120', 'X12', 'X119', 
                   'X13', 'X118', 'X114', 'X113', 'X112', 'X111', 'X110', 'X109', 'X108', 
                   'X106', 'X105', 'X103', 'X102', 'X115', 'X130', 'X127', 'X132', 'X22', 
                   'X21', 'X20', 'X19', 'X18', 'X17', 'X16', 'X131', 'X15', 'X23', 'X14', 
                   'X140', 'X138', 'X133', 'X139', 'X134', 'X135', 'X24', 'X136', 'X137']
    
    X = data[feature_cols]
    y = data['HF']
    
    results = run_optimization(X, y, model)

