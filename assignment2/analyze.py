import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import copy
if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument('task', choices=['train', 'test'], help='train or evaluate the model on a test set')
    # p.add_argument('--embeddings_path', help='path to embeddings file', required=True)
    # p.add_argument('--model_path', help='path to load/save the model')
    # p.add_argument('--baseline_strategy', choices=["most_frequent", "uniform", "stratified"],
    #                help='strategy for "dummy" classifier')
    # args = p.parse_args()

    # if args.task == 'train':
    #     train(args.embeddings_path, args.baseline_strategy, args.model_path)
    #
    # elif args.task == 'test':
    #     if args.model_path:
    models = ['models/tagger-contextual-first_token.pkl',
              'models/tagger-contextual-last_token.pkl',
              'models/tagger-contextual-reduce_max.pkl',
              'models/tagger-contextual-reduce_mean.pkl',
              'models/tagger-contextual-reduce_sum.pkl',
              'models/tagger-contextual-reduce_median.pkl',
              'models/tagger-contextual-maxmin.pkl',
              'models/tagger-contextual-random.pkl',
              'models/tagger-static.pkl',
              'models/baseline_dev.pkl']

    embeddings = ['data/en_ewt-ud-embeds-first_token.pkl',
                  'data/en_ewt-ud-embeds-last_token.pkl',
                  'data/en_ewt-ud-embeds-reduce_max.pkl',
                  'data/en_ewt-ud-embeds-reduce_mean.pkl',
                  'data/en_ewt-ud-embeds-reduce_sum.pkl',
                  'data/en_ewt-ud-embeds-reduce_median.pkl',
                  'data/en_ewt-ud-embeds-maxmin.pkl',
                  'data/en_ewt-ud-embeds-random.pkl',
                  'data/en_ewt-ud-embeds-static.pkl',
                  None]
    model_names = ['First token',
                   'Last token',
                   'Maxpool',
                   'Average',
                   'Sum',
                   'Median',
                   'Maxmin',
                   'Random'
                   'Static',
                   'Baseline']

    #models = ['models/tagger-contextual-reduce_mean.pkl', 'models/baseline_dev.pkl']
    #embeddings = ['data/en_ewt-ud-embeds-reduce_mean.pkl', None]
    #model_names = ['Average', 'Baseline']

    split = 'VALIDATION'
    #plot_models(models, embeddings, 'precision', split='VALIDATION')
    precisions = []
    recalls = []
    f1s = []
    all_tags = None
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    wrong_tokens_model = dict()
    for model_path, embedding_path in zip(models, embeddings):
        print('> Loading model from', model_path)

        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        if embedding_path==None:
            predictions_, labels_, tokens, left_context, right_context = model['predictions'], model['labels'], model['tokens'], model['left_context'], model['right_context']
        else:

            print('> Evaluating model on dataset', embedding_path)

            with open(embedding_path, "rb") as fin:
                #data, labels_ = pickle.load(fin)[split]
                data, labels_, tokens, left_context, right_context = pickle.load(fin)[split]

            predictions = model['classifier'].predict(data)
            predictions_ = model['label_encoder'].inverse_transform(
                predictions)  # inverse transform to strings for printing

        wrong_tokens = dict()
        for i in range(len(predictions_)):
            if predictions_[i] != labels_[i]:
                if tokens[i] not in wrong_tokens:
                    wrong_tokens[tokens[i]] = 1
                else:
                    wrong_tokens[tokens[i]] += 1

        wrong_tokens_norm = copy.deepcopy(wrong_tokens)
        for token in wrong_tokens:
            token_count = tokens.count(token)
            wrong_tokens_norm[token] = wrong_tokens_norm[token]/token_count
        print(dict(sorted(wrong_tokens.items(), key=lambda item: item[1], reverse=True)))
        print(dict(sorted(wrong_tokens_norm.items(), key=lambda item: item[1], reverse=True)))

        precision = metrics.precision_score(labels_, predictions_, average='macro')
        recall = metrics.recall_score(labels_, predictions_, average='macro')
        f1 = metrics.f1_score(labels_, predictions_, average='macro')

        if all_tags == None:
            all_tags = list(set(predictions_))
            tag_precision, tag_recall, tag_f1, support = metrics.precision_recall_fscore_support(labels_, predictions_,
                                                                                                 labels=all_tags,
                                                                                                 average=None)
            all_tags = [x for _, x in sorted(zip(support, all_tags))]

        tag_precision, tag_recall, tag_f1, support = metrics.precision_recall_fscore_support(labels_, predictions_, labels=all_tags, average=None)

        x = np.arange(len(all_tags))
        #ax1.scatter(x, tag_precision, s=20)#, c='b')
        ax1.plot(x, tag_precision, marker='x')  # , c='b')
        ax2.plot(x, tag_recall, marker='x')
        ax3.plot(x, tag_f1, marker='x')

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    plt.figure(1)
    plt.xticks(x, all_tags,rotation=45)
    plt.xlabel('POS tags')
    plt.ylabel('Precision')
    plt.legend(model_names)
    plt.savefig('plots/tags_precision.png', dpi=800)


    plt.figure(2)
    plt.xticks(x, all_tags, rotation=45)
    plt.xlabel('POS tags')
    plt.ylabel('Recall')
    plt.legend(model_names)
    plt.savefig('plots/tags_recall.png', dpi=800)

    plt.figure(3)
    plt.xticks(x, all_tags, rotation=45)
    plt.xlabel('POS tags')
    plt.ylabel('F1-score')
    plt.legend(model_names)
    plt.savefig('plots/tags_f1score.png', dpi=800)

    plt.figure(4)

    x = np.arange(len(model_names))


    ax4.scatter(x, precisions, s=20, c='b')
    ax4.scatter(x, f1s, s=20, c = 'c')
    ax4.scatter(x, recalls, s=20, c='r')
    plt.xticks(x, model_names)

    plt.xlabel('Model')
    plt.legend(['Precision','F1-score','Recall'])
    plt.savefig('plots/comb_type.png', dpi=800)
    plt.show()

