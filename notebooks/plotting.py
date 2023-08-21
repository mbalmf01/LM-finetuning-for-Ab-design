import seaborn as sns
from scipy.stats import pearsonr

def true_vs_pred(y_val, y_pred):
    #set style
    sns.set_style("whitegrid")
    
    #create plot
    plt.figure(figsize=(8, 6))
    sns.regplot(y=y_val, x=y_pred, scatter_kws={'s': 50, 'edgecolor': 'black', 'linewidth': 0.5}, line_kws={"color": "red"}, ci=None, scatter=True)
    
    #add titles
    plt.title('True values vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    
    #fit a line of best fit to the data
    coefficients = np.polyfit(y_pred, y_val, 1)  #1 is the degree of the polynomial (i.e., a straight line)
    line_function = np.poly1d(coefficients)
    x_line = np.linspace(min(y_pred), max(y_pred), 100)
    y_line = line_function(x_line)
    plt.plot(x_line, y_line)
    
    #add Pearson's correlation coefficient
    correlation_coefficient, _ = pearsonr(y_val, y_pred)
    plt.text(0.1, 4.8, f"Pearson's r: {correlation_coefficient:.3f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.grid(True)
    plt.show()

def regression_metrics(x_lab: str, x_list: list, reports: list):
    
    r2 = [report['R^2 Score'] for report in reports]
    mae = [report['Mean Absolute Error (MAE)'] for report in reports]
    rmse = [report['Root Mean Squared Error (RMSE)'] for report in reports]
    corr = [report['Pearson Correlation coefficient'] for report in reports]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    #Plot data on first y-axis
    line1, = ax1.plot(x_list, r2, '-o', color='green', label='R^2')
    line2, = ax1.plot(x_list, corr, '-o', color='red', label='Pearson')
    
    ax1.set_xlabel(x_lab)
    ax1.set_ylabel('R^2 score/Pearson correlation coefficient', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    #link two data
    ax2 = ax1.twinx()
    
    #plotting data on the second y-axis
    line3, = ax2.plot(batch_sizes, rmse, '-o', color='blue', label='RMSE')
    line4, = ax2.plot(batch_sizes, mae, '-s', color='orange', label='MAE')
    ax2.set_ylabel('Error Values', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    #combining legends into one box
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.grid(False)
    plt.show()