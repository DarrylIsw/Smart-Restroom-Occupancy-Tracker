<?php

namespace App\Http\Controllers;

use App\Models\UsageLog;
use Illuminate\Http\Request;

class UsageLogController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $usageLogs = UsageLog::all();
        return view('usagelogs.index', compact('usageLogs'));
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $request->validate([
            'toilet_id' => 'required|exists:toilets,id',
            'usage_count' => 'required|integer',
        ]);

        UsageLog::create($request->only('toilet_id', 'usage_count'));
        return response('Usage log created successfully!', 200);
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
