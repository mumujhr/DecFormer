import torch
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score


def eval_f1(pred, label, num_classes):
    micro = multiclass_f1_score(pred, label, num_classes=num_classes, average='micro')
    macro = multiclass_f1_score(pred, label, num_classes=num_classes, average='macro')
    return micro.item(), macro.item()


def co_train(model, data, label, split_index, optimizer):
    model.train()
    optimizer.zero_grad()
    pred1, pred2 = model(data.graph['node_feat'], data.graph['edge_index'], data.graph['calibration_mask'])
    loss = model.loss(pred1, pred2, label, split_index['train'])
    loss.backward()
    optimizer.step()
    # eval
    model.eval()
    with torch.no_grad():
        pred1, pred2 = model(data.graph['node_feat'], data.graph['edge_index'], data.graph['calibration_mask'])

        # pred1 = F.log_softmax(pred1, dim=-1)
        # pred2 = F.log_softmax(pred2, dim=-1)

        y = data.label.squeeze()
        num_classes = y.max() + 1

        y1_ = torch.argmax(pred1, dim=1).squeeze()
        micro_val1, macro_val1 = eval_f1(y1_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test1, macro_test1 = eval_f1(y1_[split_index['test']], y[split_index['test']], num_classes)

        y2_ = torch.argmax(pred2, dim=1).squeeze()
        micro_val2, macro_val2 = eval_f1(y2_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test2, macro_test2 = eval_f1(y2_[split_index['test']], y[split_index['test']], num_classes)

    return micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2


def test(model, data, split_index):
    model.eval()
    with torch.no_grad():
        pred = model(data.graph['node_feat'], data.graph['edge_index'], data.graph['calibration_mask'])
        y_hat_val = torch.argmax(pred[split_index['valid']], dim=1)
        acc_val = torch.mean(torch.eq(y_hat_val, data.label[split_index['valid']]).float())
        y_hat_test = torch.argmax(pred[split_index['test']], dim=1)
        acc_test = torch.mean(torch.eq(y_hat_test, data.label[split_index['test']]).float())

    return acc_val.item(), acc_test.item()


def co_test(model, data, split_index):
    model.eval()
    with torch.no_grad():
        pred1, pred2 = model(data.graph['node_feat'], data.graph['edge_index'], data)

        # pred1 = F.log_softmax(pred1, dim=-1)
        # pred2 = F.log_softmax(pred2, dim=-1)

        y = data.label.squeeze()
        num_classes = y.max() + 1

        y1_ = torch.argmax(pred1, dim=1).squeeze()
        micro_val1, macro_val1 = eval_f1(y1_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test1, macro_test1 = eval_f1(y1_[split_index['test']], y[split_index['test']], num_classes)

        y2_ = torch.argmax(pred2, dim=1).squeeze()
        micro_val2, macro_val2 = eval_f1(y2_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test2, macro_test2 = eval_f1(y2_[split_index['test']], y[split_index['test']], num_classes)

    return micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2
