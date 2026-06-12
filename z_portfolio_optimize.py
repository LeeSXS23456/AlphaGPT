import pandas as pd
import numpy as np
import os
from functools import reduce
from scipy import stats
import cvxpy as cp
from datetime import datetime
from collections import defaultdict
import pickle

srcdir = "/mnt/e/SJTU/intern/gtht/barra/data_base"
#adir = "/mnt/e/SJTU/intern/gtht/barra因子/result"
#spedir = "/mnt/e/SJTU/intern/gtht/barra/result"
#model_dir = "/mnt/e/SJTU/intern/gtht/barra、/data_base/lgb_models"
mcp_dict = pd.read_pickle(f"{srcdir}/stk_mcp/全A_freemcp_25_26D_dict.pkl")
#alpha_dict = pd.read_pickle(f"{adir}/延迟alpha/ortho_delay_measures_2024_2026_dict.pkl")
#return_dict = pd.read_pickle(f"{srcdir}/stk_ret/全A_ret_24_2603D_1030minute_s-1_dict.pkl")
df_return = pd.read_parquet("./data/cache/oos_predictions.parquet")
trdates = pd.read_pickle(f"{srcdir}/trading_dates.pkl")


print("Optimizing！")
def orthogonalize_pair(df, col1, col2, standardize=False):
    x1 = df[col1].values
    x2 = df[col2].values
    # 去均值（等价于带截距回归）
    x1_mean = x1.mean()
    x2_mean = x2.mean()
    x1_c = x1 - x1_mean
    x2_c = x2 - x2_mean
    # OLS beta
    beta1 = np.dot(x1_c, x2_c) / np.dot(x1_c, x1_c)
    beta0 = x2_mean - beta1 * x1_mean
    # 残差（正交部分）
    x2_res = x2 - beta0 - beta1 * x1
    df.loc[:, col2] = x2_res
    return df

#num_ori = 100
start_dt = "2025-01-02"
end_dt = "2026-03-25"
alpha_name = "MomentumVolatilityRange"#"D1_orth"
tr_filter_op = [d for d in trdates if (d >= start_dt) and (d <= end_dt)]#["2026-01-27","2026-01-28"]#[d for d in trdates if (d >= start_dt) and (d <= end_dt)]
pair_lst = [['earnings_yield', 'liquidity'],['beta', 'residual_volatility'],['non_linear_size', 'size']]
rou_list = [-0.308,0.357,0.510] #根据pair_lst去查数
with open(f"{srcdir}/index_component_日频/000905.XSHG_20_26D_dict.pkl", 'rb') as f:
    weight_dict = pickle.load(f) #series

ret_dict = defaultdict(list)#{}
dual_dict = defaultdict(list) #测试lam取值的时候改！
error_dt = defaultdict(list)
group_ret_hist = []
lam_lst = []
reg_lst = []


for idx,dt in enumerate(tr_filter_op):
    if idx == 0:
        continue
    #明确时间域
    alpha_dt_tp = pd.to_datetime(dt)
    #ret_dt_tp = alpha_dt_tp + pd.Timedelta(hours=10) + pd.Timedelta(minutes=30)
    barra_dt = dt #tr_filter_op[idx-1] [由于此时alpha需要当天盘后才知道，因此ret是下一天，因此barra就是当天]
    barra_dt_tp = pd.to_datetime(barra_dt)
    print(f"在{dt}天早上决策，10：30买入，第二天10：30卖出")

    #读取数据，最优化
    df_ret = df_return[df_return["date"] == alpha_dt_tp][["stock_id","y_true","y_pred"]].set_index("stock_id") #alpha_dict[alpha_dt_tp][["stock_id",alpha_name]].set_index("stock_id").loc[:,[alpha_name]].reset_index()
    #df_ret.name = "ret"
    df_ret = df_ret.rename(columns={"y_true": "ret"})
    #df_alpha = alpha_dict[alpha_dt_tp]
    
    df_barra = pd.read_pickle(f"{srcdir}/barra_data/whole_mkt/{barra_dt}.pkl") #000905标准化2
    X_center = pd.read_pickle(f"{srcdir}/barra_data/000905标准化3_含行业/{barra_dt}.pkl") #
    variance_frq = pd.read_pickle(f"{srcdir}/fac_ret_cov/{barra_dt}.pkl")
    variance_rq = pd.read_pickle(f"{srcdir}/spe_ret_cov/{barra_dt}.pkl")
    weight_index = weight_dict[barra_dt_tp]

    #统一顺序
    dfs = [df_ret,df_barra]
    df = df_ret.merge(df_barra,left_on="stock_id",right_on="order_book_id",how="inner")
    #df = reduce(lambda left, right: pd.merge(left, right, on="order_book_id", how="inner"), dfs)
    df = df.merge(weight_index,on="order_book_id",how="left")
    weight_bmk = df["weight"].fillna(0)

    
    # === Step 3: 预测当前 Rhat ===
    # model_path = f"{model_dir}/{alpha_name}/{dt}.pkl"
    # with open(model_path, "rb") as f:
    #     model = pickle.load(f)
    Rhat = df["y_pred"]#model.predict(df[[alpha_name]])
    
    ###微调
    Num = len(df)
    # Rhat = np.zeros(Num)

    orth_order = [x for x in variance_frq.index.tolist()[:11] if x != "comovement"]#["beta","momentum","size","non_linear_size","residual_volatility","liquidity","book_to_price","earnings_yield","growth","leverage"]
    ind_order = list(variance_frq.columns[11:].values)
    X_original = df[orth_order+ind_order].values #风格+行业
    stk_order = df["order_book_id"].tolist()
    # #部分正交化
    flat_pair = [item for sublist in pair_lst for item in sublist]
    non_orth = [var for var in orth_order if var not in flat_pair] #['book_to_price', 'growth', 'leverage', 'momentum']
    num_non = len(non_orth) #non_orth + ind_order / non_orth
    X_center_f =X_center.set_index("order_book_id").loc[stk_order, non_orth + flat_pair] #ind_order + 
    X_center_ind = X_center.set_index("order_book_id").loc[stk_order, ind_order].values #行业
    #X_center_bb = X_center.copy()
    for c1, c2 in pair_lst:
        X_center_f = orthogonalize_pair(X_center_f, c1, c2)
    
    X_center = X_center_f.values
    # X_center = X_center.set_index("order_book_id").loc[stk_order,orth_order].values #风格

    # w_m = np.sqrt(df_reg.free_mkp.values)
    # w_m = w_m / w_m.sum()
    #X_orth = weighted_orthogonize(X_ori,w_m)
    #F_cov = variance_frq.loc[orth_order + ind_order, orth_order + ind_order].values#orthogonized_factor_cov(X_orth,w_m)
    #D_diag = variance_rq.reindex(df_reg["code"]).values.ravel()
    #sqrtD = np.sqrt(D_diag)

    #根据barra做进一步完善
    F_cov_raw = variance_frq.loc[orth_order + ind_order, orth_order + ind_order].values
    diag = np.diag(np.diag(F_cov_raw))
    F_cov = 0.9 * F_cov_raw + 0.1 * diag
    D_diag = variance_rq.reindex(df["order_book_id"]).values.ravel()
    lower = np.percentile(D_diag, 1)
    upper = np.percentile(D_diag, 99)
    D_diag = np.clip(D_diag, lower, upper)
    sqrtD = np.sqrt(D_diag)
    w0 = np.ones(len(D_diag)) / len(D_diag)
    risk = w0 @ X_original @ F_cov @ X_original.T @ w0 + w0 @ np.diag(D_diag) @ w0
    ret  = np.mean(np.abs(Rhat))
    lam0 = ret / risk
    lam_lst.append(lam0)
    print(f"{dt}选择的lam：{lam0}")

    #R_cov = X_orth @ F_cov @ X_orth.T
    #设置权重向量的初始值【根据因子信号/等权】
    # w_ori = np.zeros_like(Rhat)
    # idx = np.argsort(Rhat)[::-1][:num_ori]
    # w_ori[idx] = np.exp(-np.arange(num_ori)/20)
    # w_ori /= w_ori.sum()

    print(f"\n=======  开始最优化 {dt} 组合  |  {datetime.now()}  =======\n")
    #turnover = 0.005
    w = cp.Variable(Num)
    w.value = np.zeros(Num) #np.zeros_like(Rhat)
    Xo = X_original.T @ w
    Xp = X_center.T @  (w + weight_bmk)
    Xi = X_center_ind.T @ (w + weight_bmk)
    lam = cp.Parameter(nonneg=True,value=lam0) #lam0
    #penalty = cp.sum_squares(cp.pos(Xp - x_max)) + cp.sum_squares(cp.pos(x_min - Xp))
    #gamma = cp.Parameter(nonneg=True)
    
    objective = cp.Minimize(
            lam * (cp.quad_form(Xo, F_cov) + cp.sum_squares(cp.multiply(sqrtD, w))) - cp.sum(cp.multiply(Rhat, w)) #+ gamma*penalty
        )

    for l_val in [0.01,0.1,0.3,0.5,1]:#[0.01,0.1,0.3,0.5,1]:
        #gamma.value = l_val
        x_min =[-l_val] * num_non #[-0.1] * len(ind_order) + #['book_to_price', 'growth', 'leverage', 'momentum']
        x_max = [l_val] * num_non
        #print(f"放松了{non_orth[1]}")
        # 特定因子约束设置
        specific_factor_bounds={}
        # specific_factor_bounds = {
        #    'earnings_yield': 1,       # 固定约束值
        #     'beta': 0.01,               # 固定约束值
        #     'residual_volatility': 0.01 # 基础值，会乘以 sqrt(1 - rou^2)
        # }

        for i, (c1, c2) in enumerate(pair_lst):
            current_rou2 = (rou_list[i])**2  # 取出当前配对的 rou
            if c1 in specific_factor_bounds:
                print("c1:",c1)
                base_val = specific_factor_bounds[c1]
                x_min.append(-base_val) #不想改变beta #-base_val
                x_max.append(base_val)
                if c2 in specific_factor_bounds:
                    print("c2:",c2)
                    base_val2 = specific_factor_bounds[c2]
                    x_min.append(-base_val2 * np.sqrt((1 - current_rou2)))
                    x_max.append(base_val2 * np.sqrt((1 - current_rou2)))
                else:
                    x_min.append(-l_val * np.sqrt((1 - current_rou2)))
                    x_max.append(l_val*  np.sqrt((1 - current_rou2)))
            else:
                x_min.append(-l_val),x_min.append(-l_val * np.sqrt((1 - current_rou2)))
                x_max.append( l_val),x_max.append( l_val * np.sqrt((1 - current_rou2)))

        # x_min = np.full(X_center.shape[1], -l_val) #偏离 个标准差
        # x_max = np.full(X_center.shape[1], l_val)
        constraints = [
            cp.sum(w+weight_bmk) == 1,
            w + weight_bmk >= 0,
            #w <= 0.01,
            # #cp.abs(w - w0) <= turnover,
            Xp >= x_min,
            Xp <= x_max,

            # Xi >= [-0.01] * len(ind_order),
            # Xi <= [0.01] * len(ind_order)
        ]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP,max_iter=50000,warm_start=True,verbose=True) #solver=cp.OSQP, #verbose=True #max_iter=10000,

            if w.value is None or np.any(np.isnan(w.value)):
                print(f"{dt}权重为空{prob.status}，等权买入")
                error_dt[l_val].append(f"{prob.status}_{dt}")
                w_opt = np.zeros(Num)
            else:
                dual_dict[l_val].append([constraints[i].dual_value for i in range(len(constraints))])
                w_opt = w.value

        except:
            print(f"{dt}最优化失败，等权买入")
            error_dt[l_val].append(dt)
            w_opt = np.zeros(Num)
    

        R_true = df.ret.values
        Rp_realized = w_opt @ R_true
        #ret_dict[dt]=Rp_realized
        ret_dict[l_val].append(Rp_realized)

        #存储w_opt
        df[f"w_opt_{l_val}"] = w_opt
    df_mcp = mcp_dict[alpha_dt_tp][["free_circulation"]]
    df = df.merge(df_mcp,on="order_book_id",how="left")
    df.index = [dt] * len(df) 


    # df.to_csv(f"{spedir}/组合优化/lgb_持仓信息/{alpha_name}_barra正交/w_opt持仓信息_{alpha_name}_{dt}.csv",index=True)
    # print(f"完成最优化 | {datetime.now()}")


#检查组合优化失败天数
print([len(error_dt[l_val]) for l_val in [0.01,0.1,0.3,0.5,1]])
print(error_dt)


#绘制净值走势图
import matplotlib.pyplot as plt
import random

df_500 = pd.read_excel(f"{srcdir}/000905_SH.xlsx")
df_500.index = df_500["日期"].dt.strftime('%Y-%m-%d')
df_500.rename({"涨跌幅":"000905"},axis=1,inplace=True)

ret_df = pd.DataFrame(ret_dict,index=tr_filter_op[1:])
ret_df = ret_df.merge(df_500[["000905"]],left_index=True,right_index=True)
#ret_df = pd.DataFrame(ret_dict.items(),  # 键值对columns=["date", "value"]  # 列名你可以随便改).set_index("date")
temp = pd.DataFrame(0, index=["2025-01-01"], columns=ret_df.columns)
ret0_df = pd.concat([temp, ret_df], ignore_index=False)
ret_cum = (1 + ret0_df).cumprod()
# ret_cum.insert(0,0,0.9) #选
# ret_df.insert(0,0,0.9) #选

plt.plot(ret_cum,label=ret_cum.columns)
step = max(1, len(ret_cum) // 10)  # 最多显示10个刻度
plt.xticks(
    ticks=range(0, len(ret_cum), step),  # 按步长取刻度
    labels=ret_cum.index[::step],       # 对应标签
    rotation=45,                        # 标签倾斜45度（关键！）
    fontsize=10
)
plt.grid(alpha=0.3)  # 加网格更美观
plt.legend(loc='best')
plt.tight_layout()  # 自动适配布局，防止标签截断


#保存净值及回测结果
#简单回测一下业绩
rf = 0.015
res = []
ret_cum.dropna(how='any',inplace=True)
for i in range(ret_cum.shape[1]):
    port_nav = ret_cum.iloc[:,i]
    cum_ret = port_nav.iloc[-1] / port_nav.iloc[0] - 1
    ann_ret = (cum_ret + 1)**(252/len(port_nav)) - 1 #daily freq
    ann_vol = ret_df.iloc[:,i].std() * np.sqrt(252)
    sp = (ann_ret - rf) / ann_vol
    maxd = min(port_nav / port_nav.cummax()) - 1
    km = (ann_ret - rf) / abs(maxd) 
    result_dict = {
    '累计收益率': f'{float(cum_ret):.2%}',
    '年化收益率': f'{float(ann_ret):.2%}',
    '年化波动率': f'{float(ann_vol):.2%}',
    '夏普比率': f'{sp:.2f}',
    '最大回撤': f'{maxd:.2%}',
    '卡玛比率': f'{km:.2f}'}
    res.append(pd.Series(result_dict,name=ret_cum.columns[i]))

file_path = f"./data/processed/portfolio_optimize/minmax_不同std_净值_{alpha_name}.xlsx"
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    # 第一个表：ret_cum → Sheet1
    ret_cum.to_excel(writer, sheet_name='净值', index=True)
    
    # 第二个表：backtest → Sheet2（你要的）
    pd.concat(res,axis=1).to_excel(writer, sheet_name='回测结果', index=True, header=True)

print("Excel 保存成功！两个工作表都已写入 ✅")
    
