# Create your views here.
from django.http import JsonResponse
from django.shortcuts import render

from .graph_generator import *
from .tools import *
from .static import *
from .weighting import *
import multiprocessing
from functools import partial
import pandas as pd

cores = 25
stability_process = 0
risk_bounds_process = 0

def index(request):
    request.session.clear_expired()
    return render(request, 'learning/index.html')

def draw_graph(request):
    if request.method == 'POST':
        graphs = []
        # only draw the first graph
        drawed_graph = None
        graph_type = request.POST.get('graph_type', None)
        n_vertices = int(request.POST.get('n_vertices', None))
        barabasi_m = int(request.POST.get('barabasi_m', None))
        g = GraphType.get(graph_type)(n_nodes=n_vertices, m=barabasi_m)
        request.session['graph'] = nx.node_link_data(g)
        drawed_graph = draw_sampled_graph(g)

        power_law_data = plot_power_law(drawed_graph)
        response = {'drawed_graph': nx.node_link_data(drawed_graph), 'power_law': power_law_data}
        return JsonResponse(response, safe=True)

def generate_data(request):
    if request.method == 'POST':
        data_type = request.POST.get('data_type', None)
        n_visible_features = int(request.POST.get('n_visible_features', None))
        json_graph = request.session.get('graph', default=None)
        g = nx.node_link_graph(json_graph)
        if g is None:
            return JsonResponse(None)
        X, Y = DataType.get(data_type)(g, 10).dataset(n_visible=n_visible_features)

        dataset = get_header(X.shape[1]) + np.hstack((X, Y.reshape(-1, 1))).tolist()

        dataset = {'dataset': dataset}

        return JsonResponse(dataset, safe=True)

def stability_process_get(request):
    return JsonResponse({'process': stability_process})

def stability_experiments(request):
    if request.method == 'POST':
        graph_type = request.POST.get('graph_type', None)
        n_vertices = str(request.POST.get('n_vertices', None))
        barabasi_m = int(request.POST.get('barabasi_m', None))
        data_type = request.POST.get('data_type', None)
        n_visible_features = int(request.POST.get('n_visible_features', None))
        n_experiments = int(request.POST.get('n_experiments', None))
        weighting_type = '0'
        n_vertices_list = sorted(n_vertices.split(','))
        pool = multiprocessing.Pool(cores)
        list_times = range(n_experiments)
        global stability_process
        stability_process = 0
        results = []
        for n in n_vertices_list:
            n = int(n)
            train_g = GraphType.get(graph_type)(n, barabasi_m)
            test_g = generate_complete_graph(300)
            weights = WeightingType.get(weighting_type)(train_g.nodes, train_g.edges)
            results.append(pool.map(
                    partial(single_experiment, train_g=train_g, test_g=test_g, origin_weights=weights,
                        GraphDataSet=DataType.get(data_type),
                        Model=StabilityModelType.get(data_type), n_visible_features=n_visible_features, features=10),
                    list_times))
            stability_process += 1
        pool.close()
        pool.join()
        converted_results = convert_results(results, barabasi_m, n_vertices_list, n_visible_features)
        equal_gens = []
        fmn_gens = []
        for result in results:
            equal_gen = []
            fmn_gen = []
            for r in result:
                equal_gen.append(r[0] - r[1])
                fmn_gen.append(r[2] - r[3])
            equal_gens.append(equal_gen)
            fmn_gens.append(fmn_gen)

        equal_gens_mean = np.asarray(equal_gens, dtype=np.float32)
        equal_gens_mean = np.mean(equal_gens_mean, axis=1)
        fmn_gens_mean = np.asarray(fmn_gens, dtype=np.float32)
        fmn_gens_mean = np.mean(fmn_gens_mean, axis=1)
        return JsonResponse({'results': converted_results, 'equal': equal_gens, 'fmn': fmn_gens, 'equal_mean': equal_gens_mean.tolist(), 'fmn_mean': fmn_gens_mean.tolist()}, safe=True)

def risk_bounds_experiments(request):
    if request.method == 'POST':
        graph_type = request.POST.get('graph_type', None)
        n_vertices = str(request.POST.get('n_vertices', None))
        n_visible_features = int(request.POST.get('n_visible_features', None))
        n_experiments = int(request.POST.get('n_experiments', None))
        weighting_type = request.POST.get('weighting_type', None)
        n_vertices_list = sorted(n_vertices.split(','), key=lambda x: int(x))
        pool = multiprocessing.Pool(cores)
        list_times = range(n_experiments)
        global risk_bounds_process
        risk_bounds_process = 0
        results = []
        for n in n_vertices_list:
            n = int(n)
            train_g = GraphType.get(graph_type)(n)
            test_g = generate_complete_graph(300)
            weights = WeightingType.get(weighting_type)(train_g.nodes, train_g.edges)
            results.append(pool.map(
                    partial(single_experiment, train_g=train_g, test_g=test_g, origin_weights=weights,
                        GraphDataSet=DataType.get('0'),
                        Model=RiskBoundsModelType.get('0'), n_visible_features=n_visible_features, features=10),
                    list_times))
            risk_bounds_process += 1
        pool.close()
        pool.join()
        converted_results = convert_results(results, 0, n_vertices_list, n_visible_features)
        equal_tests = []
        fmn_tests = []
        for result in results:
            equal_test = []
            fmn_test = []
            for r in result:
                equal_test.append(r[0])
                fmn_test.append(r[2])
            equal_tests.append(equal_test)
            fmn_tests.append(fmn_test)

        equal_tests_mean = np.asarray(equal_tests, dtype=np.float32)
        equal_tests_mean = np.mean(equal_tests_mean, axis=1)
        fmn_tests_mean = np.asarray(fmn_tests, dtype=np.float32)
        fmn_tests_mean = np.mean(fmn_tests_mean, axis=1)
        return JsonResponse({'results': converted_results, 'equal': equal_tests, 'fmn': fmn_tests, 'equal_mean': equal_tests_mean.tolist(), 'fmn_mean': fmn_tests_mean.tolist()}, safe=True)

def convert_results(results, barabasi_m, n_vertices_list, n_visible):
    res = [['测试误差', '训练误差', '类型', 'Barabasi图参数', '可见特征数', '顶点数']]
    for i in range(len(n_vertices_list)):
        for r in results[i]:
            res.append([r[0], r[1], '平均加权', barabasi_m, n_visible, n_vertices_list[i]])
            res.append([r[2], r[3], '分数匹配加权', barabasi_m, n_visible, n_vertices_list[i]])
    return res

def single_experiment(time, train_g, test_g, origin_weights, GraphDataSet,
        Model, n_visible_features, features=10):
    random = np.random.RandomState(current_microsecond() + time)
    w = random.uniform(-1, 1, size=features)
    b = 0  # random.normal(-1, 1)
    train_graph = GraphDataSet(train_g, features, random=random, w=w,
            b=b)
    test_graph = GraphDataSet(test_g, features, random=random, w=w,
            b=b)
    X_train, Y_train = train_graph.dataset(n_visible=n_visible_features)
    X_test, Y_test = test_graph.dataset(n_visible=n_visible_features)
    matching_weights = origin_weights
    equal_acc, equal_train_acc = Model(X_train, Y_train, X_test, Y_test)
    fmn_acc, fmn_train_acc = Model(X_train, Y_train, X_test, Y_test, matching_weights)
    return equal_acc, equal_train_acc, fmn_acc, fmn_train_acc

def init_lock(l):
    global lock
    lock = l

def get_header(n):
    header = []
    for i in range(n):
        header.append('feature%s' % i)
    header.append('label')
    return [header]
